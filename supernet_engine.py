import sys
import math
import time
import json
import random
import numpy as np
from collections import Counter
from typing import Iterable, Optional

import torch
from torch import nn

from lib import utils
from timm.data import Mixup
from timm.utils import ModelEma
from timm.utils.model import unwrap_model


def seg_criterion(inputs, target,  num_classes: int = 6):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def sample_configs(choices):
    config = {}
    config['depth']=[]
    config['embed_dim'] = []
    config['mlp_ratio']=[]
    config['num_heads'] = []
    config['kernel_size'] = []
    config['conv_choice'] = []

    for i in range(10):
        conv_choice = random.choice(choices['conv_choice'])
        config['conv_choice'].append(conv_choice)
        
    config['pool_scale'] = [1, 2, 3, 6]
    for i in range(4):
        if i > 0:
            pool_scale = config['pool_scale'][i] - random.randint(0,1)
            config['pool_scale'][i] = pool_scale

    dimensions = ['mlp_ratio', 'num_heads','kernel_size']
    if choices['super_embed_dim'][0] == 32:
        divisor = 32
    else:
        divisor = 64
    
    for i in range(len(choices['super_depth'])):
        if i==0:
            depth = choices['super_depth'][i]+ random.choice(choices['depth'])
            embed_dim = choices['super_embed_dim'][i]
            for dimension in dimensions:
                if dimension == 'mlp_ratio':
                    temp=[]
                    for j in range(depth):
                        mlp_ratio = choices['super_mlp_ratio'][i] + random.choice(choices[dimension])
                        if mlp_ratio == 0:
                            mlp_ratio = 1
                        temp.append(mlp_ratio)
                    config[dimension].append(temp)
                elif dimension == 'num_heads':
                    config[dimension].append([embed_dim//divisor for _ in range(depth)])
                    # config[dimension].append([embed_dim//32 for _ in range(depth)])
                elif dimension == 'kernel_size':
                    kernels = []
                    for k in range(depth):
                        selection = random.randint(0, 2)
                        if selection == 0:
                            kernel = 1
                        elif selection == 1:
                            kernel = 0
                        else:
                            kernel = choices['super_kernel_size'][i] + random.choice(choices[dimension])
                        kernels.append(kernel)
                    config[dimension].append(kernels)
        else:
            depth = choices['super_depth'][i] + random.choice(choices['depth'])
            embed_dim = choices['super_embed_dim'][i] + random.choice(choices['embed_dim'])
            for dimension in dimensions:
                if dimension == 'mlp_ratio':
                    temp=[]
                    for j in range(depth):
                        mlp_ratio = choices['super_mlp_ratio'][i] + random.choice(choices[dimension])
                        if mlp_ratio == 0:
                            mlp_ratio = 1
                        temp.append(mlp_ratio)
                    config[dimension].append(temp)
                elif dimension == 'num_heads':
                    config[dimension].append([embed_dim//divisor for _ in range(depth)])
                    # config[dimension].append([embed_dim//32 for _ in range(depth)])
                elif dimension == 'kernel_size':
                    kernels = []
                    for k in range(depth):
                        selection = random.randint(0, 2)
                        if selection == 0:
                            kernel = 1
                        elif selection == 1:
                            kernel = 0
                        else:
                            kernel = choices['super_kernel_size'][i] + random.choice(choices[dimension])
                        kernels.append(kernel)
                    config[dimension].append(kernels)
        config['depth'].append(depth)
        config['embed_dim'].append(embed_dim)
        # config['upsample_dim'].append(upsample_dim)
    
    return config

def train_one_epoch(model: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None, 
                    num_classes: int = 6):
    model.train()
    # seg_criterion.train()

    # set random seed
    random.seed(epoch)

    # debug
    retrain_config = {}
    retrain_config = {}
    retrain_config['depth'] = [1, 3, 3, 2]
    retrain_config['mlp_ratio'] = [[7.5], [8.0, 8.0, 8.0], [4.5, 4.0, 4.0], [3.5, 3.5]]
    retrain_config['num_heads'] = [[1], [2, 2, 3], [6, 6, 5], [7, 7]]   
    retrain_config['kernel_size'] = [[3], [5, 0, 0], [0, 5, 1], [0, 1]] 
    retrain_config['embed_dim'] = [64, 128, 320, 512] 
    retrain_config['conv_choice'] = [1, 3, 1, 1, 0, 1, 0, 1, 3, 3]
    retrain_config['pool_scale'] = [1, 2, 3, 6]


    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # sample random config
        if mode == 'super':
            config = sample_configs(choices=choices)
            # config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
            parameters = model_module.get_sampled_params_numel(config)
            print("sampled model parameters: {}".format(parameters))

        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1/2 * seg_criterion(outputs, targets, num_classes) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(samples)
                    loss = seg_criterion(outputs, targets, num_classes)
        else:
            outputs = model(samples)
            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * seg_criterion(outputs, targets, num_classes) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = seg_criterion(outputs, targets, num_classes)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, num_classes=6, 
             choices=None, mode='super', retrain_config=None):
    # criterion = torch.nn.CrossEntropyLoss()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # retrain_config = {}
    # retrain_config = {}
    # retrain_config['depth'] = [1, 3, 3, 2]
    # retrain_config['mlp_ratio'] = [[7.5], [8.0, 8.0, 8.0], [4.5, 4.0, 4.0], [3.5, 3.5]]
    # retrain_config['num_heads'] = [[1], [2, 2, 3], [6, 6, 5], [7, 7]]   
    # retrain_config['kernel_size'] = [[3], [5, 0, 0], [0, 5, 1], [0, 1]] 
    # retrain_config['embed_dim'] = [64, 128, 320, 512] 
    # retrain_config['conv_choice'] = [1, 3, 1, 1, 0, 1, 0, 1, 3, 3]
    # retrain_config['pool_scale'] = [1, 2, 3, 6]

    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices)
        # config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
        # parameters = model_module.get_sampled_params_numel(config)
        # print("sampled model parameters: {}".format(parameters))
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)


    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))
    
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()
    return confmat
   

def train_one_epoch_lat(model: torch.nn.Module, history_list, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None, 
                    num_classes: int = 6):
    model.train()
    
    # set random seed
    random.seed(epoch)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
        # print(config)
        # print(model_module.get_sampled_params_numel(config))

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # sample random config
        if mode == 'super':
            config = sample_configs(choices=choices)
            # model_vector = config2vector(config)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1/2 * seg_criterion(outputs, targets, num_classes) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    t_start = time.time()
                    outputs = model(samples)
                    # Here we calculate the inference time of a whole batch, instead of a single image
                    elapsed_time = time.time() - t_start
                    loss = seg_criterion(outputs, targets, num_classes)
                    if utils.is_main_process:
                        if epoch < 800:
                            record = {
                                'arch': config, 
                                'latency': elapsed_time, 
                            }
                            history_list.append(record)
                    
        else:
            outputs = model(samples)
            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * seg_criterion(outputs, targets, num_classes) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = seg_criterion(outputs, targets, num_classes)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
           
        else:
            loss.backward()
            optimizer.step()
            
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_lat(data_loader, model, device, num_classes=4, 
             choices=None, mode='super', retrain_config=None):
    # criterion = torch.nn.CrossEntropyLoss()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)

    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))
    
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())
        confmat.reduce_from_all_processes()
        
    return confmat


def build_latency_prior(history_path, top_k=300, smooth_beta=0.9):
    with open(history_path, 'r') as f:
        history = json.load(f)

    history = sorted(history, key=lambda x: x['latency'])[:top_k]

    # === Encoder (structured by stage and block) ===
    encoder_op_counts = []  # List[stage][block] = Counter
    for h in history:
        ks_stage = h['arch']['kernel_size']
        depth_stage = h['arch']['depth']
        for stage_idx, (ks_list, depth) in enumerate(zip(ks_stage, depth_stage)):
            # TODO
            while len(encoder_op_counts) <= stage_idx:
                encoder_op_counts.append([])
            for block_idx in range(depth):
                # Ensure Counter exists
                if len(encoder_op_counts[stage_idx]) <= block_idx:
                    encoder_op_counts[stage_idx].append(Counter())
                op = ks_list[block_idx]
                encoder_op_counts[stage_idx][block_idx][op] += 1

    encoder_op_priors = []
    for stage in encoder_op_counts:
        stage_priors = []
        for cnt in stage:
            total = sum(cnt.values())
            prior = {}
            all_ops = [0, 1, 3]  # MLP, Transformer, CNN
            for op in all_ops:
                raw_p = cnt.get(op, 0) / total if total > 0 else 1.0 / len(all_ops)
                smooth_p = smooth_beta * raw_p + (1 - smooth_beta) * (1.0 / len(all_ops))
                prior[op] = smooth_p
            total_p = sum(prior.values())
            for op in prior:
                prior[op] /= total_p
            stage_priors.append(prior)
        encoder_op_priors.append(stage_priors)

    # === Decoder: conv_choice remains unchanged ===
    decoder_blocks = [h['arch']['conv_choice'] for h in history]
    num_decoder_blocks = len(decoder_blocks[0])
    decoder_op_counts = [Counter() for _ in range(num_decoder_blocks)]
    for conv_seq in decoder_blocks:
        for i, op in enumerate(conv_seq):
            decoder_op_counts[i][op] += 1

    decoder_op_priors = []
    for cnt in decoder_op_counts:
        total = sum(cnt.values())
        prior = {}
        all_ops = list(range(5))
        for op in all_ops:
            raw_p = cnt.get(op, 0) / total if total > 0 else 1.0 / len(all_ops)
            smooth_p = smooth_beta * raw_p + (1 - smooth_beta) * (1.0 / len(all_ops))
            prior[op] = smooth_p
        total_p = sum(prior.values())
        for op in prior:
            prior[op] /= total_p
        decoder_op_priors.append(prior)

    return encoder_op_priors, decoder_op_priors


def sample_from_block_priors_decoder(block_op_priors):
    arch = []
    for prior in block_op_priors:
        ops = list(prior.keys())
        probs = np.array(list(prior.values()))
        probs /= probs.sum()
        op = np.random.choice(ops, p=probs)
        arch.append(op)
    return arch