import os
import random

import numpy as np
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader

import transforms as T

class XRayDataset(torch.utils.data.Dataset):
    """
    FasterRCNN-compatible dataset of images and annotations from the vinbigdata folder.
    """
    def __init__(self, root, transforms=None, img_size=512):
        self.root = root
        self.transforms = transforms
        self.img_size = img_size

        df = pd.read_csv(f'../vinbigdata/{root}.csv')
        # class ID 0 = aortic enlargement, remove all other annotations
        df = df[df.class_id == 0]
        df['image_path'] = f'../vinbigdata/{root}/'+df.image_id+'.png'
        df = XRayDataset.normalize_bboxes(df)
        self.df = df

    @staticmethod
    def normalize_bboxes(df):
      df['x_min'] = df.apply(lambda row: (row.x_min)/row.width, axis =1)
      df['y_min'] = df.apply(lambda row: (row.y_min)/row.height, axis =1)

      df['x_max'] = df.apply(lambda row: (row.x_max)/row.width, axis =1)
      df['y_max'] = df.apply(lambda row: (row.y_max)/row.height, axis =1)

      df['x_mid'] = df.apply(lambda row: (row.x_max+row.x_min)/2, axis =1)
      df['y_mid'] = df.apply(lambda row: (row.y_max+row.y_min)/2, axis =1)

      df['w'] = df.apply(lambda row: (row.x_max-row.x_min), axis =1)
      df['h'] = df.apply(lambda row: (row.y_max-row.y_min), axis =1)

      df['area'] = df['w']*df['h']
      return df

    def __getitem__(self, idx):
      img_df = self.df.iloc[idx]
      img_path = img_df.image_path
      img = Image.open(img_path).convert("RGB")

      # get bounding box coordinates for each mask
      boxes = []

      # already normalized, so mupliply by img size
      xmin = img_df.x_min * self.img_size
      xmax = img_df.x_max * self.img_size
      ymin = img_df.y_min * self.img_size
      ymax = img_df.y_max * self.img_size
      boxes.append([xmin, ymin, xmax, ymax])

      # convert everything into a torch.Tensor
      boxes = torch.as_tensor(boxes, dtype=torch.float32)
      # there is only one class
      labels = torch.ones((1,), dtype=torch.int64)

      image_id = torch.tensor([idx])
      area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
      # suppose all instances are not crowd
      iscrowd = torch.zeros((1,), dtype=torch.int64)

      target = {}
      target["boxes"] = boxes
      target["labels"] = labels
      target["image_id"] = image_id
      target["area"] = area
      target["iscrowd"] = iscrowd

      if self.transforms is not None:
          img, target = self.transforms(img, target)

      return img, target

    def __len__(self):
      return len(self.df.index)

def collate_fn(batch):
    return tuple(zip(*batch))

def data_loaders(args, original_dataset=None, test_dataset=None):
    if original_dataset is None:
        original_dataset = XRayDataset('train', transforms=get_transform(True))
    dataset_train, dataset_valid, dataset_test = datasets(args, original_dataset, test_dataset)

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
    loader_test = DataLoader(
        dataset_test,
        batch_size=1,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
        collate_fn=collate_fn,
    )

    return loader_train, loader_valid, loader_test

def datasets(args, original_dataset, test_dataset=None):
    all_train = original_dataset
    torch.manual_seed(1)
    random.seed(1)
    indices = torch.randperm(len(all_train)).tolist()

    # train, valid, test
    splits = [0.7, 0.1, 0.2]

    train_split_index = int(splits[0] * len(all_train))
    valid_split_index = int((splits[0] + splits[1]) * len(all_train))
    train = torch.utils.data.Subset(all_train, indices[:train_split_index])
    valid = torch.utils.data.Subset(all_train, indices[train_split_index:valid_split_index])

    if test_dataset is None:
        test_dataset = XRayDataset('train', transforms=get_transform(False))
    test = torch.utils.data.Subset(test_dataset, indices[valid_split_index:])

    print(f'train: {len(train)}, valid: {len(valid)}, test: {len(test)}')
    return train, valid, test

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
