import argparse
import datetime
import time
import os
import sys

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
import torchvision

from dataset import XRayDataset

import transforms as T

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch.utils.tensorboard import SummaryWriter

sys.path.append('..')
from early_stopping import EarlyStopping

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(args, device):
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet18', True)
    backbone.out_channels = 256
    anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)),
                                   aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
    model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
    return model

def data_loaders(args):
    dataset_train, dataset_valid, dataset_test = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
        collate_fn=collate_fn,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=1,
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

def save_batch_image(images, targets):
    fig, axs = plt.subplots(nrows=len(images), figsize=(7, len(images) * 7))
    axs = axs.flatten()

    for i in range(len(images)):
      ax = axs[i]
      ax.imshow(images[i][0].cpu().numpy())
      bbox = targets[i]['boxes'][0].cpu().numpy()
      rect = patches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
      ax.add_patch(rect)
    plt.savefig('image.png')

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
        
        #save_batch_image(images, targets)

    print("Training epoch " + str(epoch) + " complete")
    
    writer.add_scalar("Loss/train", losses, epoch)
    
@torch.no_grad()
def evaluate(model, data_loader, device, epoch=None, writer=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in tqdm(data_loader):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        coco_evaluator.update(res)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    mAP = coco_evaluator.coco_eval['bbox'].stats[1]
    if writer is not None:
      writer.add_scalar("Loss/valid", mAP, epoch)

    print("Validation complete")
    return coco_evaluator

def main(args):
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    loader_train, loader_valid = data_loaders(args)

    model = get_model(args, device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    early_stopping = EarlyStopping(patience=10)
    start_epoch = 0
    
    if os.path.exists(f'runs/{args.experiment_name}'):
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
        train_one_epoch(model, optimizer, loader_train, device, epoch, 10, writer)
        lr_scheduler.step()

        eval = evaluate(model, loader_valid, device, epoch, writer)
        mAP = eval.coco_eval['bbox'].stats[0]
        early_stopping(-mAP)
        if early_stopping.early_stop:
            print("Stopping early")
            break

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
          }, f'runs/{args.experiment_name}/checkpoint.pt')

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