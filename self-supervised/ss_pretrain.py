import os
import sys
import glob
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss

sys.path.append('..')
from dataset import XRayDataset, datasets

class SimCLRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, 1
        )
        return [optim], [scheduler]
    
def ss_pretrain(args, backbone, input_size, num_ftrs=32 * 16, batch_size=128, max_epochs=20):
  num_workers = 8
  seed = 1

  pl.seed_everything(seed)

  collate_fn = lightly.data.SimCLRCollateFunction(
      input_size=input_size,
      vf_prob=0.5,
      rr_prob=0.5
  )

  # We create a torchvision transformation for embedding the dataset after
  # training
  test_transforms = torchvision.transforms.Compose([
      torchvision.transforms.Resize((input_size, input_size)),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(
          mean=0,
          std=1,
      ),
      torchvision.transforms.ToPILImage()
  ])

  # TODO Make sure the perm and split is identical to baseline and repeatable
  all_train = lightly.data.dataset.LightlyDataset('../vinbigdata/train')
  indices = torch.randperm(len(all_train)).tolist()
  split_index = int(0.01 * len(all_train))
  dataset_train = torch.utils.data.Subset(all_train, indices[:split_index])
  dataset_valid = torch.utils.data.Subset(all_train, indices[split_index:])

  dataloader_train_simclr = torch.utils.data.DataLoader(
      dataset_train,
      batch_size=batch_size,
      shuffle=True,
      collate_fn=collate_fn,
      drop_last=True,
      num_workers=num_workers
  )

  dataloader_valid = torch.utils.data.DataLoader(
      dataset_valid,
      batch_size=batch_size,
      shuffle=False,
      drop_last=False,
      num_workers=num_workers
  )

  gpus = 1 if torch.cuda.is_available() else 0

  model = SimCLRModel()
  device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
  model.to(device)
  trainer = pl.Trainer(
    max_epochs=1, gpus=gpus, progress_bar_refresh_rate=100
  )
  trainer.fit(model, dataloader_train_simclr)
  pretrained_resnet_backbone = model.backbone

  # you can also store the backbone and use it in another code
  state_dict = {
      'resnet18_parameters': pretrained_resnet_backbone.state_dict()
  }
  torch.save(state_dict, 'pretrained-backbone.pth')
  return model.backbone
