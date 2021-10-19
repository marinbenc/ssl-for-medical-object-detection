import os
import numpy as np
import torch
from PIL import Image
import pandas as pd

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
          img = self.transforms(img)

      return img, target

    def __len__(self):
      return len(self.df.index)