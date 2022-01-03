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
import torchvision

sys.path.append('..')
from dataset import XRayDataset

import transforms as T

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

import train
from train import get_untrained_model, data_loaders

def main(args):
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    loader_train, _, loader_test = data_loaders(args)

    model = get_untrained_model(args, device)
    model.to(device)

    files = os.listdir(f'runs/{args.experiment_name}')
    files.sort()
    checkpoints = [f for f in files if '.pt' in f]
    if len(checkpoints) > 0:
      checkpoint = torch.load(f'runs/{args.experiment_name}/{checkpoints[-1]}')
      model.load_state_dict(checkpoint['model'])
      model.eval()

    evaluate(model, loader_test, device)

def evaluate(model, loader_test, device):
    coco_metrics = train.baseline_train.evaluate(model, loader_test, device)
    mAP_50_to_95 = coco_metrics.coco_eval['bbox'].stats[0]

    return

    predictions = []
    gts = []
    for image, targets in tqdm(loader_test):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(image)
        outputs = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in outputs]
        predictions.append(outputs)
        gts.append(targets)

    for (pred, gt) in zip(predictions, gts):
        # TODO calculate stuff
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validation'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='number of workers for data loading (default: 4)',
    )    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--experiment-name', type=str, 
        default=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), 
        help='the name of the experiment'
    )
    args = parser.parse_args()
    main(args)