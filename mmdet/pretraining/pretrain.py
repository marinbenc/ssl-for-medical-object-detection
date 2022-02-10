import os
import sys
import glob
import argparse
import datetime
import shutil

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

import lightly
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss

import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np
import pandas

from mmdet.core.export import build_model_from_cfg
from mmdet.models import build_backbone, build_neck
from mmdet.datasets import build_dataset, build_dataloader

sys.path.append('..')
from base_config import get_config
import train

class HistogramNormalize:
  """Performs histogram normalization on numpy array and returns 8-bit image.

  Code was taken and adapted from Facebook:
  https://github.com/facebookresearch/CovidPrognosis

  """

  def __init__(self, number_bins: int = 256):
    self.number_bins = number_bins

  def __call__(self, image: np.array) -> Image:
    image = np.array(image)[:, :, 0]
    # get image histogram
    image_histogram, bins = np.histogram(
        image.flatten(), self.number_bins, density=True
    )
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    return Image.fromarray(image_equalized.reshape(image.shape))

class GaussianNoise:
  """Applies random Gaussian noise to a tensor.

  The intensity of the noise is dependent on the mean of the pixel values.
  See https://arxiv.org/pdf/2101.04909.pdf for more information.

  """

  def __call__(self, sample: torch.Tensor) -> torch.Tensor:
    mu = sample.mean()
    snr = np.random.randint(low=4, high=8)
    sigma = mu / snr
    noise = torch.normal(torch.zeros(sample.shape), sigma)
    return sample + noise

class SimCLRModel(pl.LightningModule):
  def __init__(self, backbone, hidden_dim, lr):
    super().__init__()
    self.lr = lr
    self.backbone = nn.Sequential(*list(backbone.children()))
    self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)
    self.criterion = NTXentLoss()

  def forward(self, x):
    h = self.backbone(x)[-1].flatten(start_dim=1)
    z = self.projection_head(h)
    return z

  def training_step(self, batch, batch_idx):
    (x0, x1), _, _ = batch
    z0 = self.forward(x0)
    z1 = self.forward(x1)
    loss = self.criterion(z0, z1)
    self.log("train_loss_ssl", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    (x0, x1), _, _ = batch
    z0 = self.forward(x0)
    z1 = self.forward(x1)
    loss = self.criterion(z0, z1)
    self.log("valid_loss_ssl", loss)
    return loss

  def configure_optimizers(self):
    optim = torch.optim.SGD(
      self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optim, 1
    )
    return [optim], []

def get_backbone():
  cfg = get_config(base_directory='..')
  out_channels = cfg.model.neck.in_channels[0]
  backbone = build_backbone(cfg.model.backbone)
  return backbone, out_channels

def get_ss_model(lr):
  backbone, out_channels = get_backbone()
  return SimCLRModel(backbone, out_channels, lr=0.001)

def ss_pretrain(experiment_name, replace, batch_size, epochs, lr, workers, labeled_dataset_percent):
  num_ftrs = 32 * 16
  input_size = 512
  seed = 1
  pl.seed_everything(seed)

  transform = torchvision.transforms.Compose([
    HistogramNormalize(),
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.RandomResizedCrop(size=input_size, scale=(0.2, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.GaussianBlur(21),
    torchvision.transforms.ToTensor(),
    GaussianNoise(),
  ])

  collate_fn_train = lightly.data.BaseCollateFunction(transform=transform)

  collate_fn_test = lightly.data.BaseCollateFunction(transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.ToTensor(),
  ]))

  test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
      mean=0,
      std=1,
    ),
    torchvision.transforms.ToPILImage()
  ])

  # construct a lightly dataset with only the pretraining files
  pretrain_dataset_mmdet, _ = train.get_training_datasets(labeled_dataset_percent, base_directory='..')
  filenames = [image['file_name'].split('/')[-1] for image in pretrain_dataset_mmdet.data_infos]
  dataset_train = lightly.data.dataset.LightlyDataset('../data/train', filenames=filenames)
  print('Samples in pretrain dataset: ' + str(len(dataset_train)))

  dataset_valid = lightly.data.dataset.LightlyDataset('../data/valid')

  dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn_train,
    drop_last=True,
    num_workers=workers
  )

  dataloader_valid = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn_test,
    num_workers=workers
  )

  gpus = 1 if torch.cuda.is_available() else 0
  model = get_ss_model(lr)
  device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
  model.to(device)

  output_folder = "../../vinbig_output/" + experiment_name
  if os.path.exists(output_folder):
    if replace:
      shutil.rmtree(output_folder + '/pretrain')
      os.makedirs(output_folder + '/pretrain')
    else:
      print("Folder already exists.")
      return
    
  logger = TensorBoardLogger(output_folder + '/pretrain')

  trainer = pl.Trainer(
    max_epochs=epochs, gpus=gpus, progress_bar_refresh_rate=100, logger=logger
  )

  batch = next(iter(dataloader_train_simclr))
  (augmentation_1, augmentation_2), _, _ = batch
  for i in range(len(augmentation_1)):
    img1 = augmentation_1[i]
    img2 = augmentation_2[i]
    img1 = img1.squeeze().cpu().numpy()[0]
    img2 = img2.squeeze().cpu().numpy()[0]

    fig, axs = plt.subplots(1, 2)
    axs = axs.flatten()
    axs[0].imshow(img1)
    axs[1].imshow(img2)
    plt.show()

  trainer.fit(model, dataloader_train_simclr, dataloader_valid)
  pretrained_resnet_backbone = model.backbone

  state_dict = {
    'resnet18_parameters': pretrained_resnet_backbone.state_dict()
  }
  torch.save(state_dict, output_folder + '/pretrain/pretrained-backbone.pth')
  #show_neighbors(model, dataloader_valid, device)
  return model.backbone

def show_neighbors(model, dataloader_test, device):
    model.eval()
    embeddings, fnames = generate_embeddings(model, dataloader_test, device)
        # transform the original bounding box annotations to multiclass labels
    fnames = [fname.split('.')[0] for fname in fnames]

    df = pandas.read_csv('../../vinbigdata/train.csv')
    classes = list(np.unique(df.class_name))
    filenames = list(np.unique(df.image_id))

    # iterate over all bounding boxes and add a one-hot label if an image contains
    # a bounding box of a given class, after that, the array "multilabels" will
    # contain a row for every image in the input dataset and each row of the
    # array contains a one-hot vector of critical findings for this image
    multilabels = np.zeros((len(dataloader_test), len(classes)))
    for filename, label in zip(df.image_id, df.class_name):
        try:
            i = fnames.index(filename.split('.')[0])
            j = classes.index(label)
            multilabels[i, j] = 1.
        except Exception:
            pass

    # plot the distribution of the multilabels of the k nearest neighbors of
    # the three example images at index 4111, 3340, 1796
    k = 10
    plot_knn_multilabels(
        embeddings, multilabels, range(0, 20), fnames, classes, n_neighbors=k
    )

def generate_embeddings(model, dataloader, device):
    """Generates representations for all images in the dataloader"""

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, label, fnames in dataloader:
            img = img[0]
            img = img.to(device)
            model.backbone.to(device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb.detach().cpu())
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames

def plot_knn_multilabels(
    embeddings, multilabels, samples_idx, filenames, classes, n_neighbors=50
):
    """Plots multiple rows of random images with their nearest neighbors"""
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)

    # position the bars
    bar_width = 0.4
    r1 = np.arange(multilabels.shape[1])
    r2 = r1 + bar_width

    # loop through our randomly picked samples
    for idx in samples_idx:
        fig = plt.figure()

        bars1 = multilabels[idx]
        bars2 = np.mean(multilabels[indices[idx]], axis=0)

        plt.title(filenames[idx])
        plt.bar(r1, bars1, color='steelblue', edgecolor='black', width=bar_width)
        plt.bar(r2, bars2, color='lightsteelblue', edgecolor='black', width=bar_width)
        plt.xticks(0.5 * (r1 + r2), classes, rotation=90)
        plt.tight_layout()
        plt.show()

def parse_args():
  parser = argparse.ArgumentParser(
    description='Pretraining, saves the pretrained backbone checkpoint as a file'
  )
  parser.add_argument(
      '--batch-size',
      type=int,
      default=4,
      help='input batch size for training',
  )
  parser.add_argument(
      '--epochs',
      type=int,
      default=100,
      help='number of epochs to train',
  )
  parser.add_argument(
      '--workers',
      type=int,
      default=8,
  )
  parser.add_argument(
      '--lr',
      type=float,
      default=0.001,
      help='initial learning rate (default: 0.001)',
  )
  parser.add_argument(
      '--experiment-name', type=str, 
      default=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), 
      help='the name of the experiment'
  )
  parser.add_argument('--labeled-dataset-percent', type=float, default=1)
  parser.add_argument(
      '--replace',
      default=False,
      action='store_true',
      help='replace the experiment with a new one'
  )
  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  ss_pretrain(**vars(args))

if __name__ == '__main__':
  main()