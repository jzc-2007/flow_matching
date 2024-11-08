# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImageNet input pipeline."""

import numpy as np
import os
import random
import jax
import torch
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, MNIST
from PIL import Image

from absl import logging
from functools import partial

from models.edm.augment import AugmentPipe


IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def prepare_batch_data(batch, config, batch_size=None):
  """Reformat a input batch from PyTorch Dataloader.
  
  Args:
    batch = (image, label)
      image: shape (host_batch_size, 3, height, width)
      label: shape (host_batch_size)
    batch_size = expected batch_size of this node, for eval's drop_last=False only
  """
  image, label = batch  

  if config.aug.use_edm_aug:
    augment_pipe = AugmentPipe(p=0.12, xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
    image, augment_label = augment_pipe(image)
  else:
    augment_label = None

  # pad the batch if smaller than batch_size
  if batch_size is not None and batch_size > image.shape[0]:
    image = torch.cat([image, torch.zeros((batch_size - image.shape[0],) + image.shape[1:], dtype=image.dtype)], axis=0)
    label = torch.cat([label, -torch.ones((batch_size - label.shape[0],), dtype=label.dtype)], axis=0)
    assert augment_label is None  # don't support padding augment_label

  # reshape (host_batch_size, 3, height, width) to
  # (local_devices, device_batch_size, height, width, 3)
  local_device_count = jax.local_device_count()
  image = image.permute(0, 2, 3, 1)
  image = image.reshape((local_device_count, -1) + image.shape[1:])
  label = label.reshape(local_device_count, -1)

  image = image.numpy()
  label = label.numpy()

  if config.model.use_aug_label:
    assert config.aug.use_edm_aug
    augment_label = augment_label.reshape((local_device_count, -1) + augment_label.shape[1:])
    augment_label = augment_label.numpy()
  else:
    augment_label = None

  return_dict = {
    'image': image,
    'label': label,
    'augment_label': augment_label,
  }

  return return_dict


def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


from torchvision.datasets.folder import pil_loader
def loader(path: str):
    return pil_loader(path)


def create_split(
    dataset_cfg,
    batch_size,
    split,
):
  """Creates a split from the ImageNet dataset using Torchvision Datasets.

  Args:
    dataset_cfg: Configurations for the dataset.
    batch_size: Batch size for the dataloader.
    split: 'train' or 'val'.
  Returns:
    it: A PyTorch Dataloader.
    steps_per_epoch: Number of steps to loop through the DataLoader.
  """
  num_output_channels = dataset_cfg.out_channels

  rank = jax.process_index()
  if split == 'train':
    if dataset_cfg.root == "MNIST":
      ds = datasets.MNIST(
        root='~/cache',
        train=True,
        download=True,
        transform=transforms.Compose([
          # transforms.Resize((32, 32)),  # Resize images to 32x32 pixels
          transforms.Grayscale(num_output_channels=num_output_channels),  # Convert to three channels
          transforms.ToTensor(),
          transforms.Normalize((0.5,), (0.5,)),
        ]),
      )
    elif dataset_cfg.root == "CIFAR":
      ds = datasets.CIFAR10(
        root=dataset_cfg.data_dir,
        train=True,
        download=True,
        transform=transforms.Compose([
          # transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),),
          # transforms.RandomHorizontalFlip(p=0.5),
          # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
      )
    else:
      raise NotImplementedError
      ds = datasets.ImageFolder(
        os.path.join(dataset_cfg.root, split),
        transform=transforms.Compose([
          transforms.RandomResizedCrop(IMAGE_SIZE, interpolation=3),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB),
        ]),
        loader=loader,
      )
    logging.info(ds)
    sampler = DistributedSampler(
      ds,
      num_replicas=jax.process_count(),
      rank=rank,
      shuffle=True,
    )
    it = DataLoader(
      ds, batch_size=batch_size, drop_last=True,
      worker_init_fn=partial(worker_init_fn, rank=rank),
      sampler=sampler,
      num_workers=dataset_cfg.num_workers,
      prefetch_factor=dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None,
      pin_memory=dataset_cfg.pin_memory,
      persistent_workers=True if dataset_cfg.num_workers > 0 else False,
    )
    steps_per_epoch = len(it)
  else:
    raise NotImplementedError

  return it, steps_per_epoch
