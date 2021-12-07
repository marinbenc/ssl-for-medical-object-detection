import os
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

sys.path.insert(1, '../baseline')
import train as baseline_train

def ss_pretrain(backbone, input_size, num_ftrs=32 * 16, batch_size=128, max_epochs=20):
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
      )
  ])

  dataset_train, dataset_valid, _ = baseline_train.datasets(args)

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

  # create the SimCLR model using the newly created backbone
  model = lightly.models.SimCLR(backbone, num_ftrs=num_ftrs)

  criterion = lightly.loss.NTXentLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
  encoder = lightly.embedding.SelfSupervisedEmbedding(
      model,
      criterion,
      optimizer,
      dataloader_train_simclr
  )

  gpus = 1 if torch.cuda.is_available() else 0

  encoder.train_embedding(gpus=gpus,
                          progress_bar_refresh_rate=100,
                          max_epochs=max_epochs,
                          )

  device = 'cuda' if gpus==1 else 'cpu'
  encoder = encoder.to(device)

  embeddings, _, fnames = encoder.embed(dataloader_test, device=device)
  embeddings = normalize(embeddings)

  pretrained_resnet_backbone = model.backbone

  # you can also store the backbone and use it in another code
  state_dict = {
      'resnet18_parameters': pretrained_resnet_backbone.state_dict()
  }
  torch.save(state_dict, 'pretrained-backbone.pth')

  return model.backbone
