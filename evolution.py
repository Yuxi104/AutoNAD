import os
import json
import yaml
import time
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from lib import utils
import lib.transforms as T
from lib.seg_dataset import VOCSegmentation
from lib.config import cfg, update_config_from_file

from model.supernet import AutoNAD
from supernet_engine import evaluate_lat, build_latency_prior, sample_from_block_priors_decoder


class SegmentationPresetTrain:
    def __init__(self, dataset, hflip_prob=0.5, vflip_prob=0.5,rotate_prob=0.5,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        
        if dataset == 'mt':
            base_size = 320
            crop_size = None
            trans = [T.Resize((base_size, base_size))]
        elif dataset == 'neu':
            base_size = 200
            crop_size = None
            trans = [T.RandomResize(base_size, base_size)]
        elif dataset == 'msd':
            base_size = 540
            crop_size = None
            trans = [T.Resize(base_size)]
                
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        if rotate_prob > 0:
            trans.append(T.RandomRotate(rotate_prob))
        if crop_size is not None:
            trans.append(T.RandomCrop(crop_size))
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

class SegmentationPresetEval:
    def __init__(self, dataset, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        if dataset == 'mt':
            base_size = 320
            trans = [T.Resize((base_size, base_size))]
        elif dataset == 'neu':
            base_size = 200
            trans = [T.RandomResize(base_size, base_size)]
        elif dataset == 'msd':
            base_size = 540
            trans = [T.RandomResize(base_size)]
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_transform(train, dataset):
    return SegmentationPresetTrain(dataset=dataset) if train else SegmentationPresetEval(dataset=dataset)


def decode_cand_tuple(cand_tuple):
    depth = []
    mlp_ratio = []
    num_heads = []
    kernel_size = []
    embed_dim = []
    pool_scale = list(cand_tuple[-4:])
    conv_choice = list(cand_tuple[-14:-4])

    pointer = 0
    for _ in range(4):
        depth_temp = int(cand_tuple[pointer])
        depth.append(depth_temp)
        mlp_ratio.append(list(cand_tuple[pointer + 1:pointer + depth_temp + 1]))
        num_heads.append(list(cand_tuple[pointer + depth_temp + 1:pointer + 2 * depth_temp + 1]))
        kernel_size.append(list(cand_tuple[pointer+2*depth_temp+1:pointer+3*depth_temp+1]))
        embed_dim.append(cand_tuple[pointer+3*depth_temp+1])
        pointer = int(pointer+3*depth_temp+2)

    return depth, mlp_ratio, num_heads, kernel_size, embed_dim, conv_choice, pool_scale


class EvolutionSearcher(object):

    def __init__(self, args, device, model, model_without_ddp, 
                 choices, val_loader, output_dir, history_path):
        self.device = device
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.s_prob = args.s_prob
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.checkpoint_path = args.resume
        self.candidates = []
        self.top_miou = []
        self.top_lat = []
        self.cand_params = []
        self.choices = choices      
        self.divisor = 32

        # Build sample prior
        self.encoder_prior, self.decoder_prior = build_latency_prior(history_path)

    def save_checkpoint(self):

        info = {}
        info['top_miou'] = self.top_miou
        info['top_lat'] = self.top_lat
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        checkpoint_path = os.path.join(self.output_dir, "evo/checkpoint-{}.pth.tar".format(self.epoch))

        if utils.is_main_process():
            torch.save(info, checkpoint_path)
            print('save checkpoint to', checkpoint_path)

        # torch.save(info, checkpoint_path)
        # print('save checkpoint to', checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return False
        info = torch.load(self.checkpoint_path)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', self.checkpoint_path)
        return True
  
    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        
        depth, mlp_ratio, num_heads, kernel_size, embed_dim, conv_choice, pool_scale = decode_cand_tuple(cand)
        sampled_config = {}
        sampled_config['depth'] = depth
        sampled_config['mlp_ratio'] = mlp_ratio
        sampled_config['num_heads'] = num_heads

        sampled_config['kernel_size'] = kernel_size
        sampled_config['embed_dim'] = embed_dim
        sampled_config['conv_choice'] = conv_choice
        sampled_config['pool_scale'] = pool_scale
        

        n_parameters = self.model_without_ddp.get_sampled_params_numel(sampled_config)
        info['params'] = n_parameters / 10. ** 6

        for i in range(len(depth)):
            if not isinstance(depth[i], int):
                print(sampled_config)

        if info['params'] > self.parameters_limits:
            print('parameters limit exceed')
            return False

        if info['params'] < self.min_parameters_limits:
            print('under minimum parameters limit')
            return False

        print("rank:", utils.get_rank(), cand, info['params'])
        t_start = time.time()
        eval_stats = evaluate_lat(self.val_loader, self.model, self.device, args.nb_classes, 
                              mode='retrain',retrain_config=sampled_config)
        elapsed_time = time.time() - t_start
     
        val_info = str(eval_stats)
        miou = float(val_info[-4:])
        info['miou'] = miou
        if utils.is_main_process:
            info['latency'] = elapsed_time
        info['visited'] = True

        return True
    
    # Get the penalty term on specific device
    def get_penalty_term(self):
        latency_measurements = []
        total_samples = 80
        warmup_samples = 40
        current_iter = 0
        cand_iter = self.stack_random_cand_for_latency_penalty(self.get_random_cand)
        
        print('Start measuring benchmark latency ({} samples, {} warmup) ......'.format(
            total_samples, warmup_samples))

        while current_iter < total_samples:
            cand = next(cand_iter)
            depth, mlp_ratio, num_heads, kernel_size, embed_dim, conv_choice, pool_scale = decode_cand_tuple(cand)
            sampled_config = {}
            sampled_config['depth'] = depth
            sampled_config['mlp_ratio'] = mlp_ratio
            sampled_config['num_heads'] = num_heads
            sampled_config['kernel_size'] = kernel_size
            sampled_config['embed_dim'] = embed_dim
            sampled_config['conv_choice'] = conv_choice
            sampled_config['pool_scale'] = pool_scale
        
            t_start = time.time()
            eval_stats = evaluate_lat(self.val_loader, self.model, self.device, args.nb_classes, 
                                mode='retrain',retrain_config=sampled_config)
            elapsed_time = time.time() - t_start
        
            if current_iter >= warmup_samples:
                if utils.is_main_process:
                    latency_measurements.append(elapsed_time)
                    print('Measured latency sample {}/{}: {:.4f} s'.format(
                        len(latency_measurements), total_samples - warmup_samples, elapsed_time))
            else:
                print('Warmup sample {}/{}'.format(current_iter + 1, warmup_samples))
            current_iter += 1

            
        base_latency = np.mean(latency_measurements)
        print('\n Benchmark Latency (L_base) calculated: {:.4f} s'.format(base_latency))
        W = 5.0 
        if base_latency > 1e-6:
            self.latency_penalty = W / base_latency
        else:
            # If the latency is too small (close to zero), set a conservative penalty value
            self.latency_penalty = 0.01 

        if utils.is_main_process:
            print('Final Latency Penalty (A) set to: {:.4f} (W={:.1f})'.format(
                self.latency_penalty, W))
   
        
    def update_top_k(self, population, k):
        scored = sorted(
            [p for p in population if p in self.vis_dict],
            key=lambda x: self.vis_dict[x]['miou'] - self.latency_penalty * self.vis_dict[x]['latency'],
            reverse=True
        )
        self.keep_top_k[k] = scored[:k]


    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def stack_random_cand_for_latency_penalty(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                yield cand

    # Get cand_tuple
    def get_random_cand(self):
        cand_tuple = list()
        dimensions = ['mlp_ratio', 'num_heads', 'kernel_size']

        conv_choice = sample_from_block_priors_decoder(self.decoder_prior)
        pool_scale = [1, 2, 3, 6]
        for i in range(4):
            if i > 0:
                pool_scale_temp = pool_scale[i] - random.randint(0,1)
                pool_scale[i] = pool_scale_temp

        for i in range(len(self.choices['super_depth'])):
            depth = self.choices['super_depth'][i] + random.choice(self.choices['depth'])
            cand_tuple.append(depth)

            if i == 0:
                embed_dim = self.choices['super_embed_dim'][i]
            else:
                embed_dim = self.choices['super_embed_dim'][i] + random.choice(self.choices['embed_dim'])

            for dimension in dimensions:
                if dimension == 'mlp_ratio':
                    for _ in range(depth):
                        cand_tuple.append(self.choices['super_mlp_ratio'][i] + random.choice(self.choices[dimension]))
                elif dimension == 'num_heads':
                    for _ in range(depth):
                        cand_tuple.append(embed_dim//self.divisor)
                elif dimension == 'kernel_size':
                    for block_idx in range(depth):
                        prior = self.encoder_prior[i][block_idx]  # i is the stage index
                        ops = list(prior.keys())  # e.g. [0,1,3]
                        probs = np.array(list(prior.values()))
                        probs /= probs.sum()
                        kernel = np.random.choice(ops, p=probs)
                        cand_tuple.append(kernel)
            cand_tuple.append(embed_dim)
            
        cand_tuple.extend(conv_choice)
        cand_tuple.extend(pool_scale)

        return tuple(cand_tuple)

    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        # print(cand_iter)
        while len(self.candidates) < num:
            cand = next(cand_iter)

            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            depth, mlp_ratio, num_heads, kernels, embed_dim, conv_choice, pool_scale = decode_cand_tuple(cand)
    
            random_s = random.random()
            result_cand = []

            # depth
            if random_s < s_prob:
                for i in range(len(depth)):
                    new_depth = self.choices['super_depth'][i] + random.choice(self.choices['depth'])
                    if new_depth > depth[i]:
                        mlp_ratio[i] = mlp_ratio[i] + [(self.choices['super_mlp_ratio'][i]+random.choice(self.choices['mlp_ratio']))  for _ in
                                                 range(new_depth - depth[i])]
                        
                        num_heads[i] = num_heads[i] + [0 for _ in range(new_depth - depth[i])] # Use 0 as a placeholder.
                        kernels_temp_list = []
                        for idx in range(new_depth - depth[i]):
                            prior = self.encoder_prior[i][depth[i]+idx]  # i is the stage index
                            ops = list(prior.keys())  # e.g. [0,1,3]
                            probs = np.array(list(prior.values()))
                            probs /= probs.sum()
                            kernel = np.random.choice(ops, p=probs)
                            kernels_temp_list.append(kernel)
                        kernels[i] = kernels[i] + kernels_temp_list
                    else:
                        mlp_ratio[i] = mlp_ratio[i][:new_depth]
                        num_heads[i] = num_heads[i][:new_depth]
                        kernels[i] = kernels[i][:new_depth]
                    depth[i] = new_depth

            # mlp_ratio
            for i in range(len(depth)):
                for j in range(depth[i]):
                    random_s = random.random()
                    if random_s < m_prob:
                        mlp_ratio[i][j] = self.choices['super_mlp_ratio'][i]+random.choice(self.choices['mlp_ratio'])

            # embed_dim
            for i in range(len(depth)):
                if i==0:
                    embed_dim[i] = self.choices['super_embed_dim'][i]
                else:
                    random_s = random.random()
                    if random_s < m_prob:
                        embed_dim[i] = self.choices['super_embed_dim'][i] + random.choice(self.choices['embed_dim'])

            # num_heads
            for i in range(len(depth)):
                for j in range(depth[i]):
                    num_heads[i][j] = embed_dim[i] // self.divisor
            
            
            # kernels
            for i in range(len(depth)):
                for j in range(depth[i]):
                    random_s = random.random()
                    if random_s < m_prob:
                        prior = self.encoder_prior[i][j]
                        ops = list(prior.keys())
                        probs = np.array(list(prior.values()))
                        probs /= probs.sum()
                        kernel = np.random.choice(ops, p=probs)
                        kernels[i][j] = kernel
            
            # conv_choice
            for i in range(len(conv_choice)):
                random_s = random.random()
                if random_s < m_prob:
                    prior = self.decoder_prior[i]
                    ops = list(prior.keys())
                    probs = np.array(list(prior.values()))
                    probs /= probs.sum()
                    conv_choice_temp = np.random.choice(ops, p=probs)
                    conv_choice[i] = conv_choice_temp   

            # pool_scale
            pool_scale = [1, 2, 3, 6]
            for i in range(len(pool_scale)):
                random_s = random.random()
                if random_s < m_prob:
                    if i > 0:
                        pool_scale_temp = pool_scale[i] - random.randint(0,1)
                        pool_scale[i] = pool_scale_temp

            # result_cand = [depth] + mlp_ratio + num_heads + kernels + [embed_dim]
            for i in range(len(depth)):
                result_cand.append(depth[i])
                result_cand = result_cand + mlp_ratio[i]
                result_cand = result_cand + num_heads[i]
                result_cand = result_cand + kernels[i]
                result_cand.append(embed_dim[i])

            result_cand.extend(conv_choice)
            result_cand.extend(pool_scale)
            return tuple(result_cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            # print(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res
    
    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        # iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            depth1, mlp_ratio1, num_heads1, kernels1, embed_dim1, conv_choice1, pool_scale1 = decode_cand_tuple(p1)
            depth2, mlp_ratio2, num_heads2, kernels2, embed_dim2, conv_choice2, pool_scale2 = decode_cand_tuple(p2)
            max_iters_tmp = 50
            result_cand = []
            while max_iters_tmp > 0:
                # Four-stage randomized crossover
                index1 = random.randint(0, 3)
                index2 = random.randint(0, 3)

                # conv crossover
                cross_conv = list(random.choice([i, j]) for i, j in zip(conv_choice1, conv_choice2))
                # pool_scale crossover
                cross_pool_scale = list(random.choice([i, j]) for i, j in zip(pool_scale1, pool_scale2))


                if depth1[index1] == depth2[index1] and depth1[index2] == depth2[index2]:
                    cross_mlp1 = list(random.choice([i, j]) for i, j in zip(mlp_ratio1[index1], mlp_ratio2[index1]))
                    cross_kernels1 = list(random.choice([i, j]) for i, j in zip(kernels1[index1], kernels2[index1]))
                    cross_embed_dim1 = random.choice([embed_dim1[index1], embed_dim2[index1]])
                    cross_mlp2 = list(random.choice([i, j]) for i, j in zip(mlp_ratio1[index2], mlp_ratio2[index2]))
                    cross_kernels2 = list(random.choice([i, j]) for i, j in zip(kernels1[index2], kernels2[index2]))
                    cross_embed_dim2 = random.choice([embed_dim1[index2], embed_dim2[index2]])

                    mlp_ratio1[index1] = cross_mlp1
                    mlp_ratio1[index2] = cross_mlp2
                    kernels1[index1] = cross_kernels1
                    kernels2[index2] = cross_kernels2
                    embed_dim1[index1] = cross_embed_dim1
                    embed_dim1[index2] = cross_embed_dim2
                    for i in range(len(depth1)):
                        result_cand.append(depth1[i])
                        result_cand = result_cand + mlp_ratio1[i]

                        new_num_heads_val = embed_dim1[i] // self.divisor
                        num_heads1[i] = [new_num_heads_val] * depth1[i]
                        result_cand = result_cand + num_heads1[i]

                        result_cand = result_cand + kernels1[i]
                        result_cand.append(embed_dim1[i])

                    result_cand.extend(cross_conv)
                    result_cand.extend(cross_pool_scale)
                    return tuple(result_cand)

                if depth1[index1] == depth2[index1] or depth1[index2] == depth2[index2]:
                    if depth1[index1] == depth2[index1]:
                        index = index1
                    else:
                        index = index2
                    cross_mlp1 = list(random.choice([i, j]) for i, j in zip(mlp_ratio1[index], mlp_ratio2[index]))
                    cross_kernels1 = list(random.choice([i, j]) for i, j in zip(kernels1[index], kernels2[index]))
                    cross_embed_dim1 = random.choice([embed_dim1[index], embed_dim2[index]])

                    mlp_ratio1[index] = cross_mlp1
                    kernels1[index] = cross_kernels1
                    embed_dim1[index] = cross_embed_dim1
                    for i in range(len(depth1)):
                        result_cand.append(depth1[i])
                        result_cand = result_cand + mlp_ratio1[i]

                        new_num_heads_val = embed_dim1[i] // self.divisor
                        num_heads1[i] = [new_num_heads_val] * depth1[i]
                        result_cand = result_cand + num_heads1[i]

                        result_cand = result_cand + kernels1[i]
                        result_cand.append(embed_dim1[i])
                    result_cand.extend(cross_conv)
                    result_cand.extend(cross_pool_scale)
                    return tuple(result_cand)
                else:
                    max_iters_tmp -= 1
                    p1 = random.choice(self.keep_top_k[k])
                    p2 = random.choice(self.keep_top_k[k])
                    depth1, mlp_ratio1, num_heads1, kernels1,embed_dim1, conv_choice1, pool_scale1 = decode_cand_tuple(p1)
                    depth2, mlp_ratio2, num_heads2, kernels2, embed_dim2, conv_choice2, pool_scale2 = decode_cand_tuple(p2)
            print("keep same")
            return tuple(p1)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res
    
    # Get best architectures
    def export_deployment_models(self):

        top_model_cand = self.keep_top_k[50][0]
        models = {'Top1': top_model_cand}
        A = self.latency_penalty 
        for k, cand in models.items():
            latency = self.vis_dict[cand]['latency']
            miou = self.vis_dict[cand]['miou']
            
            score = miou - A * latency
            arch = self.decode_to_config(cand)
            print(f'Best Modle:')
            print(f'   MioU = {miou:.2f}')
            print(f'   Latency = {latency:.2f} ms')
            print(f'   Score = {score:.2f} (MioU - A * Latency, A={A:.4f})')
            print(f'   Model Configuration: {arch}')
            return arch


    def decode_to_config(self, cand):
        depth, mlp_ratio, num_heads, kernel_size, embed_dim, conv_choice, pool_scale = decode_cand_tuple(cand)
        return {
            'depth': depth, 'mlp_ratio': mlp_ratio, 'num_heads': num_heads,
            'kernel_size': kernel_size, 'embed_dim': embed_dim,
            'conv_choice': conv_choice, 'pool_scale': pool_scale
        }


    def search(self):
        self.get_penalty_term()

        print(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)
            self.update_top_k(
                self.candidates, k=self.select_num)
            self.update_top_k(
                self.candidates, k=50)
            
            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            
            tmp_miou = []
            tmp_lat = []
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} val miou = {}, params = {}, latency = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['miou'], self.vis_dict[cand]['params'], self.vis_dict[cand]['latency']))
                tmp_miou.append(self.vis_dict[cand]['miou'])
                tmp_lat.append(self.vis_dict[cand]['latency'])

            self.top_miou.append(tmp_miou)
            self.top_lat.append(tmp_lat)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob, self.s_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)
            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1
            # Update every time
            checkpoint_path = self.save_checkpoint()
            
        if utils.is_main_process:
            arch = self.export_deployment_models()
        return arch, checkpoint_path



def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)

    # evolution search parameters
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    # parser.add_argument('--select-num', type=int, default=2)
    parser.add_argument('--population-num', type=int, default=50)
    # parser.add_argument('--population-num', type=int, default=2)
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--s_prob', type=float, default=0.4)
    parser.add_argument('--crossover-num', type=int, default=25)
    # parser.add_argument('--crossover-num', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--mutation-num', type=int, default=25)
    # parser.add_argument('--mutation-num', type=int, default=2)
    parser.add_argument('--param-limits', type=float, default=23)
    parser.add_argument('--min-param-limits', type=float, default=18)

    # config file
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=14,
                        help='max distance in relative position embedding')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # custom model argument
    parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01_101/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR100', 'IMNET', 'INAT', 'INAT19', 'EVO_IMNET'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_mi', default='', help='resume from mi estimator checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.set_defaults(amp=True)

    parser.add_argument('--nb-classes', default=4, type=int)
    parser.add_argument('--base-size', default=200, type=int)
    parser.add_argument('--crop-size', default=None)
    parser.add_argument('--dataset', type=str)

    return parser


def main(args):
    update_config_from_file(args.cfg)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    # save config for later experiments
    with open(os.path.join(args.output_dir, "evo/config_evo.yaml"), 'w') as f:
        f.write(args_text)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    dataset_val = VOCSegmentation(args.data_path,
                                  transforms=get_transform(train=False,
                                                             dataset=args.dataset),
                                  aug=False,
                                  txt_name="val.txt")
    print("-------------------------------------")
    args.nb_classes = args.nb_classes + 1
    print(f"The class number (including background): {args.nb_classes}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(2 * args.batch_size),
        sampler=sampler_val, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    print(f"Creating SuperNet")
    print(cfg)
    
    model = AutoNAD(embed_dims=cfg.SUPERNET.EMBED_DIM, depths=cfg.SUPERNET.DEPTH,
                          num_heads=cfg.SUPERNET.NUM_HEADS, mlp_ratio=cfg.SUPERNET.MLP_RATIO,
                          kernel_size=cfg.SUPERNET.KERNEL_SIZE, choice=cfg.SUPERNET.CONV_CHOICE,
                          qkv_bias=True, drop_rate=args.drop,
                          drop_path_rate=args.drop_path,
                          gp=args.gp,
                          num_classes=args.nb_classes)

    choices = {'mlp_ratio': cfg.SEARCH_SPACE.MLP_RATIO, 'embed_dim': cfg.SEARCH_SPACE.EMBED_DIM, 
               'depth': cfg.SEARCH_SPACE.DEPTH,'kernel_size':cfg.SEARCH_SPACE.KERNEL_SIZE, 
               'conv_choice': cfg.SUPERNET.CONV_CHOICE, 'super_num_heads': cfg.SUPERNET.NUM_HEADS, 
               'super_mlp_ratio': cfg.SUPERNET.MLP_RATIO, 'super_embed_dim': cfg.SUPERNET.EMBED_DIM, 
               'super_depth': cfg.SUPERNET.DEPTH,'super_kernel_size': cfg.SUPERNET.KERNEL_SIZE
               }
    

    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        print("resume from checkpoint: {}".format(args.resume))
        model_without_ddp.load_state_dict(checkpoint['model'])

    t = time.time()
    history_path = os.path.join(args.output_dir, f"train_history.json")
    searcher = EvolutionSearcher(args, device, 
                                 model, model_without_ddp, 
                                 choices, data_loader_val, 
                                 args.output_dir, history_path)

    arch, checkpoint_path = searcher.search()
    print('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))
    
    load_checkpoint(checkpoint_path)

    if utils.is_main_process():
        arch_config_path = os.path.join(args.output_dir, 'evo', 'best_arch.json')
        with open(arch_config_path, 'w') as f:
            json.dump(arch, f, indent=4, cls=utils.NumpyEncoder)
        print(f"Best architecture saved to {arch_config_path}")
    

def load_checkpoint(checkpoint_path):
    if not utils.is_main_process():
        return

    if os.path.isfile(checkpoint_path):
        info = torch.load(checkpoint_path, map_location='cpu') # 建议加 map_location
        vis_dict = info['vis_dict']
        keep_top_k = info['keep_top_k']
        
        with open(os.path.join(args.output_dir, "evo/keep_top_k.txt"), 'w') as f:
            f.write(str(keep_top_k))
        with open(os.path.join(args.output_dir, "evo/vis_dict.txt"), 'w') as f:
            f.write(str(vis_dict))
        print(f"Successfully loaded and processed {checkpoint_path}")
    else:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('AutoFormer evolution search', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    evo_dir = os.path.join(args.output_dir, "evo")
    if evo_dir:
        os.makedirs(evo_dir, exist_ok=True)

    main(args)
    