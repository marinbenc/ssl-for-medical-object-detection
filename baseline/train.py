import argparse
import json
import os
import sys
import datetime
from pprint import pprint

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

import torchvision.transforms as T

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

def main(args):
    #makedirs(args)
    #snapshotargs(args)
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


    return
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # criterion =

    metrics = {
      'dsc': DiceMetric(device=device),
      'loss': Loss(criterion)
    }

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    trainer.logger = setup_logger('Trainer')

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    train_evaluator.logger = setup_logger('Train Evaluator')
    validation_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    validation_evaluator.logger = setup_logger('Val Evaluator')

    best_dsc = 0

    @trainer.on(Events.GET_BATCH_COMPLETED(once=1))
    def plot_batch(engine):
        if fold != 0:
            return

        x, y = engine.state.batch
        # show_images_row(
        #     [x_i[0].detach().cpu().squeeze().numpy() for x_i in x] + 
        #     [x_i[1].detach().cpu().squeeze().numpy() for x_i in x] + 
        #     [y_i.detach().cpu().squeeze().numpy() for y_i in y], rows=3, vmin=0., vmax=1.)
        # plt.show()

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        nonlocal best_dsc
        train_evaluator.run(loader_train)
        validation_evaluator.run(loader_valid)
        curr_dsc = validation_evaluator.state.metrics['dsc']
        if curr_dsc > best_dsc:
            best_dsc = curr_dsc

    log_dir = f'logs/{args.experiment_name}/fold_{fold}'
    tb_logger = TensorboardLogger(log_dir=log_dir)

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag='training',
        output_transform=lambda loss: {'batchloss': loss},
        metric_names='all',
    )

    for tag, evaluator in [('training', train_evaluator), ('validation', validation_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names='all',
            global_step_transform=global_step_from_engine(trainer),
        )

    def score_function(engine):
        return engine.state.metrics['dsc']

    model_checkpoint = ModelCheckpoint(
        log_dir,
        n_saved=1,
        score_function=score_function,
        filename_prefix='',
        score_name='dsc',
        global_step_transform=global_step_from_engine(trainer),
        require_empty=False
    )

    validation_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {'model': model})

    trainer.run(loader_train, max_epochs=args.epochs)
    tb_logger.close()



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