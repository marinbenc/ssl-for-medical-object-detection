import argparse
import datetime
import time
import os

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

from dataset import XRayDataset

import transforms as T

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

from train import get_model, data_loaders, evaluate

def main(args):
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    loader_train, loader_valid = data_loaders(args)

    model = get_model(args, device)
    model.to(device)

    files = os.listdir(f'runs/{args.experiment_name}')
    files.sort()
    checkpoints = [f for f in files if '.pt' in f]
    if len(checkpoints) > 0:
      checkpoint = torch.load(f'runs/{args.experiment_name}/{checkpoints[-1]}')
      model.load_state_dict(checkpoint['model'])
      model.eval()

    evaluate(model, loader_valid, device)

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
        '--experiment_name', type=str, 
        default=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), 
        help='the name of the experiment'
    )
    args = parser.parse_args()
    main(args)