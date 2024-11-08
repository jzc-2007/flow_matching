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

# Copyright 2021 The Flax Authors.
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
"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Model
  config.model = ml_collections.ConfigDict()
  config.model.image_size = 32
  config.model.out_channels = 1

  config.model.base_width = 64
  config.model.n_T = 18  # inference stepss
  config.model.dropout = 0.0

  config.model.use_aug_label = False

  config.model.net_type = 'ncsnpp'

  config.aug = ml_collections.ConfigDict()
  config.aug.use_edm_aug = False

  # Consistency training
  config.ct = ml_collections.ConfigDict()
  config.ct.start_ema = 0.9
  config.ct.start_scales = 2
  config.ct.end_scales = 150

  # Dataset
  config.dataset = dataset = ml_collections.ConfigDict()
  dataset.name = 'imagenet'
  dataset.root = '/kmh-nfs-us-mount/data/imagenet'
  dataset.num_workers = 4
  dataset.prefetch_factor = 2
  dataset.pin_memory = False
  dataset.cache = False
  dataset.out_channels = 0  # from model
  dataset.steps_per_epoch = -1
  dataset.data_dir = '/kmh-nfs-us-mount/data/cifar10'

  # Eval fid
  config.fid = ml_collections.ConfigDict()
  config.fid.num_samples = 50000
  config.fid.fid_per_epoch = 500
  config.fid.on_use = True
  config.fid.eval_only = False
  config.fid.device_batch_size = 128
  config.fid.cache_ref = '/kmh-nfs-us-mount/data/cached/cifar10_jax_stats_20240820.npz'

  # Training
  config.optimizer = 'sgd'

  config.learning_rate = 0.1
  config.lr_schedule = 'cosine'  # 'cosine'/'cos', 'const'

  config.weight_decay = 0.0001  
  config.adam_b1 = 0.9
  config.adam_b2 = 0.95

  config.warmup_epochs = 5.
  config.momentum = 0.9
  config.batch_size = 128
  config.shuffle_buffer_size = 16 * 128
  config.prefetch = 10

  config.num_epochs = 100
  config.log_per_step = 100
  config.log_per_epoch = -1
  config.eval_per_epoch = 1000
  config.visualize_per_epoch = 1
  config.checkpoint_per_epoch = 200

  config.steps_per_eval = -1

  config.restore = ''
  config.pretrain = ''

  config.half_precision = False

  config.seed = 0  # init random seed

  return config


def metrics():
  return [
      'train_loss',
      'eval_loss',
      'train_accuracy',
      'eval_accuracy',
      'steps_per_second',
      'train_learning_rate',
  ]
