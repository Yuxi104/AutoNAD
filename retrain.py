import os
import time
import json
import torch
import datetime

import lib.transforms as T
from lib.seg_dataset import VOCSegmentation
from lib.config import cfg, update_config_from_file
from model.supernet import AutoNAD
from retrain_utils import (create_lr_scheduler, save_on_master, 
                           mkdir, init_distributed_mode, 
                           evaluate_retrain, retrain_one_epoch)


class SegmentationPresetTrain:
    def __init__(self, dataset, hflip_prob=0.5, vflip_prob=0.5,rotate_prob=0.5,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        print(dataset)
        if dataset == 'mt':
            base_size = 320
            crop_size = None
            trans = [T.Resize((base_size, base_size))]
        elif dataset == 'neu':
            base_size = 200
            crop_size = None
            trans = [T.Resize((base_size, base_size))]
        elif dataset == 'msd':
            base_size = 512
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
            trans = [T.Resize((base_size, base_size))]
        elif dataset == 'msd':
            base_size = 512
            trans = [T.Resize(base_size)]

        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, dataset):
    return SegmentationPresetTrain(dataset=dataset) if train else SegmentationPresetEval(dataset=dataset)


def main(args):
    init_distributed_mode(args)
    update_config_from_file(args.cfg)
    print(args)

    device = torch.device(args.device)
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1
    results_file = "{}/{}_train_results{}.txt".format(args.output_dir, args.dataset, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = VOCSegmentation(args.data_path,
                                    transforms=get_transform(train=True,
                                                             dataset=args.dataset),
                                    aug = args.aug,
                                    txt_name="train.txt")
    if args.test:
        val_dataset = VOCSegmentation(args.data_path,
                                    transforms=get_transform(train=False,
                                                             dataset=args.dataset),
                                    aug = False,
                                    txt_name="test.txt")
    else:
        val_dataset = VOCSegmentation(args.data_path,
                                    transforms=get_transform(train=False,
                                                             dataset=args.dataset),
                                    aug = False,
                                    txt_name="val.txt")

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn, drop_last=True)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)

    print("Creating model")
    # Read the arch from json file.
    with open(args.subnet, 'r') as f:
        retrain_config = json.load(f)
    print(retrain_config)  

    
    model = AutoNAD(embed_dims=cfg.SUPERNET.EMBED_DIM, depths=cfg.SUPERNET.DEPTH,
                          num_heads=cfg.SUPERNET.NUM_HEADS, mlp_ratio=cfg.SUPERNET.MLP_RATIO, 
                          kernel_size=cfg.SUPERNET.KERNEL_SIZE,
                          qkv_bias=True, drop_rate=0.0,
                          drop_path_rate=0.1,
                          gp=True,
                          num_classes=num_classes)


    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    
    optimizer= torch.optim.Adam(model_without_ddp.parameters(),args.lr, 
                                weight_decay = args.weight_decay)
    

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs, warmup=True)


    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("loading supernet weight")

        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test:
        confmat = evaluate_retrain(model, val_data_loader, device=device, 
                                   num_classes=num_classes, retrain_config=retrain_config)
        val_info = str(confmat)
        print(val_info)
        return


    print("Start training")
    start_time = time.time()
    best_miou = 0.
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = retrain_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler,
                                        retrain_config=retrain_config)

        confmat = evaluate_retrain(model, val_data_loader, device=device, 
                                   num_classes=num_classes, retrain_config=retrain_config)
        val_info = str(confmat)
        miou = float(val_info[-4:])
        print(val_info)

        # Record the result from every epoch
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                            f"train_loss: {mean_loss:.4f}\n" \
                            f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.output_dir:
            save_file = {'model': model_without_ddp.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'lr_scheduler': lr_scheduler.state_dict(),
                         'args': args,
                         'epoch': epoch}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            if best_miou <= miou:
                best_miou = miou
                save_on_master(save_file, os.path.join(args.output_dir, f'{args.dataset}.pth'))
                print("checkpoint saved!!!!")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='/media/data3/lyx/DEFECT-VOC/NEU')
    parser.add_argument('--device', default='cuda:1', help='device')
    # number of classses (without background)
    parser.add_argument('--num-classes', default=3, type=int, help='num_classes')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using SyncBatchNorm')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./fix_train', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    # supnet config
    parser.add_argument('--cfg',default=r"")
    # best subnet config
    parser.add_argument('--subnet',default=r"")
    # Validate the result on test set
    parser.add_argument('--test',action='store_true')
    # Using extra augumentation
    parser.add_argument('--aug',action='store_true')
    parser.add_argument('--dataset',type=str)
    args = parser.parse_args()

    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
