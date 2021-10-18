import os
import numpy as np
import torch
from PIL import Image
import pandas as pd


class XRayDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, size=512):
        self.root = root
        self.transforms = transforms

        df = pd.read_csv(f'vinbigdata/{root}.csv')
        df = df[df.class_id == 0]
        df['image_path'] = f'vinbigdata/{root}/'+df.image_id+'.png'
        df = self.normalize_coords(df)

        self.imgs = 
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def normalize_coords(df):
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
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)