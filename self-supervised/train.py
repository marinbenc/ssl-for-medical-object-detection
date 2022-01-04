import argparse
import datetime
import time
import os
import sys

import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection import FasterRCNN, SSD
from torchvision.models.detection.ssd import _vgg_extractor
from torchvision.models import vgg
import torchvision

sys.path.append('..')
from dataset import XRayDataset, data_loaders, get_transform

import transforms as T

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch.utils.tensorboard import SummaryWriter

sys.path.append('..')
from early_stopping import EarlyStopping

sys.path.append('..')
import baseline.train as baseline_train
from ss_pretrain import ss_pretrain, get_ss_model

def get_untrained_model(args, device):
    backbone = get_ss_model().backbone
    backbone.to(device)
    model = get_downstream_model(backbone, args, device)
    return model

def get_downstream_model(backbone, args, device):
    # anchor_sizes = ((32, 64, 128, 256, 512),)
    # aspect_ratios = ((0.5, 1.0, 2.0),)
    # anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=7, sampling_ratio=2)
    # backbone.out_channels = 512
    # model = FasterRCNN(backbone,
    #                rpn_anchor_generator=anchor_generator,
    #                num_classes=2,
    #                box_roi_pool=roi_pooler)
    # model.to(device)

   # backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet18', True)
    model = torchvision.models.detection.retinanet_resnet50_fpn(num_classes=2)
    model.to(device)
    return model

def main(args):
    try:
        os.mkdir(f'runs/{args.experiment_name}')
    except OSError as error:
        print(error)
    with open(f'runs/{args.experiment_name}/args.json', 'w') as args_file:
        json.dump(args, args_file, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    if args.skip_pretraining:
        pretrained_model_backbone = get_ss_model().backbone
        pretrained_model_backbone.train()
        pretrained_model_backbone.to(device)
    else:
        if args.pretrained_backbone is not None:
            checkpoint = torch.load(args.pretrained_backbone)
            ss_model = get_ss_model()
            ss_model.load_state_dict(checkpoint['state_dict'])
            pretrained_model_backbone = ss_model.backbone
            pretrained_model_backbone.to(device)
        else:
            pretrained_model_backbone = train_self_supervised_pretraining(args)
    pretrained_model_backbone.out_channels = 512
    train_downstream(pretrained_model_backbone, args, device)

def train_self_supervised_pretraining(args):
    backbone = ss_pretrain(args, 512, batch_size=args.pretrain_batch_size, max_epochs=args.pretrain_epochs)
    # TODO DRY
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    backbone.to(device)
    return backbone

def train_downstream(backbone, args, device):
    model = get_downstream_model(backbone, args, device)
    loader_train, loader_valid, _ = data_loaders(args, XRayDataset('train', transforms=get_transform(False)))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    early_stopping = EarlyStopping(patience=10)
    start_epoch = 0

    if not args.replace and os.path.exists(f'runs/{args.experiment_name}'):
      files = os.listdir(f'runs/{args.experiment_name}')
      files.sort()
      checkpoints = [f for f in files if '.pt']
      if len(checkpoints) > 0:
        checkpoint = checkpoints[-1]
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        model.train()

    num_epochs = args.epochs

    writer = SummaryWriter(log_dir=f'runs/{args.experiment_name}')

    for epoch in tqdm(range(start_epoch, num_epochs)):
        baseline_train.train_one_epoch(model, optimizer, loader_train, device, epoch, 10, writer)
        lr_scheduler.step()

        # eval = baseline_train.evaluate(model, loader_valid, device, epoch, writer)
        # mAP = eval.coco_eval['bbox'].stats[0]
        # early_stopping(-mAP)
        # if early_stopping.early_stop:
        #     print("Stopping early")
        #     break

        # torch.save({
        #     'epoch': epoch,
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #   }, f'runs/{args.experiment_name}/checkpoint.pt')

    writer.flush()
    writer.close()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training'
    )
    parser.add_argument(
        '--pretrain-batch-size',
        type=int,
        default=4,
        help='input batch size for training (default: 16)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='input batch size for training (default: 16)',
    )
    parser.add_argument(
        '--pretrain-epochs',
        type=int,
        default=100,
        help='number of epochs to train (default: 100)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of epochs to train (default: 100)',
    )
    parser.add_argument(
        '--pretrain-lr',
        type=float,
        default=0.001,
        help='initial learning rate (default: 0.001)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='initial learning rate (default: 0.001)',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='number of workers for data loading (default: 4)',
    )
    parser.add_argument(
        '--experiment-name', type=str, 
        default=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), 
        help='the name of the experiment'
    )
    parser.add_argument(
        '--pretrained-backbone', type=str, 
        default=None, 
        help='the file path of the pretrained backbone'
    )
    parser.add_argument(
        '--replace',
        default=False,
        action='store_true',
        help='replace the experiment with a new one'
    )
    parser.add_argument(
        '--skip-pretraining',
        default=False,
        action='store_true',
        help='replace the experiment with a new one'
    )
    args = parser.parse_args()
    main(args)