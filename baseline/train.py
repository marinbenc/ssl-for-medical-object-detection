import argparse
import json
import os
import sys
import datetime
from pprint import pprint
import time

import numpy as np
import torch
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ignite.utils import setup_logger
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger,
    global_step_from_engine,
)
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from sklearn.model_selection import KFold

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from dataset import XRayDataset

import transforms as T

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

from torch.utils.tensorboard import SummaryWriter

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# def get_splits(args, patients):
#     kfolds = KFold(n_splits=args.folds, shuffle=False)
#     patients.sort()
#     patients = np.array(patients)
#     return patients, kfolds.split(patients)

def get_model(args, device):
  # load a model pre-trained on COCO
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  # replace the classifier with a new one, that has
  # num_classes which is user-defined
  num_classes = 2  # 1 class (person) + background
  # get number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  # replace the pre-trained head with a new one
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  
  return model

def data_loaders(args):
    dataset_train, dataset_valid, dataset_test = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
        collate_fn=collate_fn,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
        collate_fn=collate_fn,
    )

    return loader_train, loader_valid

def datasets(args):
    all_train = XRayDataset('train', transforms=get_transform(True))
    indices = torch.randperm(len(all_train)).tolist()
    split_index = int(0.75 * len(all_train))
    train = torch.utils.data.Subset(all_train, indices[:split_index])
    valid = torch.utils.data.Subset(all_train, indices[split_index:])

    #test = XRayDataset('test', transforms=get_transform(False))
    return train, valid, ''

def check_pipeline():
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    loader_train, loader_valid = data_loaders(args)

    model = get_model(args, device)
    model.to(device)

    images,targets = next(iter(loader_train))
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    output = model(images,targets)   # Returns losses and detections
    # For inference
    model.eval()
    x = [torch.rand(3, 512, 512).to(device), torch.rand(3, 512, 512).to(device)]
    predictions = model(x)           # Returns predictions
    print(predictions)
    print(output)

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, writer: SummaryWriter):
    model.train()
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        break
    
    writer.add_scalar("Loss/train", losses, epoch)
    
@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in data_loader:
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

def main(args):
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    loader_train, loader_valid = data_loaders(args)

    model = get_model(args, device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = args.epochs

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, loader_train, device, epoch, 10, writer)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        #evaluate(model, loader_valid, device=device)

    writer.flush()
    writer.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='input batch size for training (default: 16)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of epochs to train (default: 100)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='initial learning rate (default: 0.001)',
    )
    parser.add_argument(
        '--logs', type=str, default='./logs', help='folder to save logs'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='number of workers for data loading (default: 4)',
    )
    parser.add_argument(
        '--experiment_name', type=str, 
        default=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), 
        help='the name of the experiment'
    )
    args = parser.parse_args()
    main(args)