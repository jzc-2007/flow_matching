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

"""ImageNet example.

This script trains a ResNet-50 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

import functools
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
import jax
from jax import lax
import jax.numpy as jnp
from jax import random
import ml_collections
import optax
import numpy as np
import os

import input_pipeline
from input_pipeline import prepare_batch_data
import models.models_ddpm as models_ddpm
from models.models_ddpm import generate, edm_ema_scales_schedules

import utils.writer_util as writer_util  # must be after 'from clu import metric_writers'
from utils.info_util import print_params
from utils.vis_util import make_grid_visualization, visualize_cifar_batch
from utils.ckpt_util import restore_checkpoint, restore_pretrained, save_checkpoint
from utils.frozen_util import extract_trainable_parameters, merge_params
from utils.state_utils import flatten_state_dict
from utils.trainstate_util import TrainState

import utils.fid_util as fid_util
import utils.sample_util as sample_util
from utils import logging_util


NUM_CLASSES = 10


def create_model(*, model_cls, half_precision, **kwargs):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return model_cls(num_classes=NUM_CLASSES, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
  fake_bz = 2
  input_shape = (fake_bz, image_size, image_size, model.out_channels)
  label_shape = (fake_bz,)

  @jax.jit
  def init(*args):
    return model.init(*args)

  logging.info('Initializing params...')
  variables = init({'params': key, 'gen': key, 'dropout': key}, jnp.ones(input_shape, model.dtype), jnp.ones(label_shape, jnp.int32))
  if 'batch_stats' not in variables:
    variables['batch_stats'] = {}
  logging.info('Initializing params done.')
  return variables['params'], variables['batch_stats']


def cross_entropy_loss(logits, labels):
  one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
  return jnp.mean(xentropy)


def compute_metrics(dict_losses):
  metrics = dict_losses.copy()
  metrics = lax.all_gather(metrics, axis_name='batch')
  metrics = jax.tree_map(lambda x: x.flatten(), metrics)  # (batch_size,)
  return metrics


def create_learning_rate_fn(
    config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int,
):
  """Create learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=0.0,
      end_value=base_learning_rate,
      transition_steps=config.warmup_epochs * steps_per_epoch,
  )
  if config.lr_schedule in ['cosine', 'cos']:
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    sched_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
  elif config.lr_schedule == 'const':
    sched_fn = optax.constant_schedule(base_learning_rate)
  else:
    raise NotImplementedError
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, sched_fn],
      boundaries=[config.warmup_epochs * steps_per_epoch],
  )
  return schedule_fn


def train_step(state, batch, rng_init, learning_rate_fn, ema_scales_fn, config):
  """Perform a single training step."""

  # ResNet has no dropout; but maintain rng_dropout for future usage
  rng_step = random.fold_in(rng_init, state.step)
  rng_device = random.fold_in(rng_step, lax.axis_index(axis_name='batch'))
  rng_gen, rng_dropout = random.split(rng_device)

  trainable_params, frozen_params = get_trainable(state.params, config)

  ema_decay, scales = ema_scales_fn(state.step)

  def loss_fn(params_to_train):
    """loss function used for training."""
    
    # merge
    params = merge_params(params_to_train, frozen_params)

    outputs, new_model_state = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        imgs = batch['image'],
        train = True,
        mutable=['batch_stats'],
        rngs=dict(gen=rng_gen, dropout=rng_dropout),
    )
    loss, dict_losses, images = outputs

    return loss, (new_model_state, dict_losses, images)

  step = state.step
  dynamic_scale = state.dynamic_scale
  lr = learning_rate_fn(step)

  if dynamic_scale:
    raise NotImplementedError
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # aux, grads = grad_fn(state.params)
    aux, grads = grad_fn(trainable_params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = lax.pmean(grads, axis_name='batch')

  # for simplicity, we don't all gather images
  new_model_state, dict_losses, images = aux[1]
  metrics = compute_metrics(dict_losses)
  metrics['lr'] = lr

  new_state = state.apply_gradients(
      grads=grads, params=trainable_params, batch_stats=new_model_state['batch_stats']
  )

  # -------------------------------------------------------
  # handle ema:
  params = {
    'net': new_state.params['net'],
    'net_ema': jax.tree_map(
        lambda old, new: old * ema_decay + new * (1.0 - ema_decay),
        new_state.params['net_ema'],
        new_state.params['net'],)
  }
  new_state = new_state.replace(params=params)
  metrics['ema_decay'] = ema_decay
  metrics['scales'] = scales
  # -------------------------------------------------------

  # -------------------------------------------------------
  # sanity
  # ema_outputs, _ = state.apply_fn(
  #     {'params': {'net': new_state.params['net_ema'],
  #                 'net_ema': new_state.params['net_ema'],},
  #      'batch_stats': state.batch_stats},
  #     batch['image'],
  #     batch['label'],
  #     mutable=['batch_stats'],
  #     rngs=dict(gen=rng_gen),
  # )
  # _, ema_dict_losses, _ = ema_outputs
  # ema_metrics = compute_metrics(ema_dict_losses)

  # metrics['ema_loss_train'] = ema_metrics['loss_train']
  # metrics['delta_loss_train'] = metrics['loss_train'] - ema_metrics['loss_train']
  # -------------------------------------------------------

  return new_state, metrics, images


def sample_step(params, sample_idx, model, rng_init, device_batch_size):
  """
  sample_idx: each random sampled image corrresponds to a seed
  """
  rng_sample = random.fold_in(rng_init, sample_idx)  # fold in sample_idx
  images = generate(params, model, rng_sample, n_sample=device_batch_size)

  images_all = lax.all_gather(images, axis_name='batch')  # each device has a copy  
  images_all = images_all.reshape(-1, *images_all.shape[2:])
  return images_all


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  if state.batch_stats == {}:
    return state
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def get_trainable(params, config):
  # seperate the ema network
  trainable_prefixes = ['net/',]
  trainable_params, frozen_params = extract_trainable_parameters(params, trainable_prefixes)

  # sanity check
  assert list(frozen_params.keys()) == ['net_ema']
  assert jax.tree_structure(trainable_params['net']) == jax.tree_structure(frozen_params['net_ema'])
  return trainable_params, frozen_params


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, learning_rate_fn
):
  """Create initial training state."""
  dynamic_scale = None
  platform = jax.local_devices()[0].platform
  if config.half_precision and platform == 'gpu':
    dynamic_scale = dynamic_scale_lib.DynamicScale()
  else:
    dynamic_scale = None

  params, batch_stats = initialized(rng, image_size, model)

  # overwrite the ema net initialization
  params['net_ema'] = jax.tree_map(lambda x: jnp.array(x), params['net'])
  assert batch_stats == {}  # we don't handle this in ema

  logging.info("Info of params['net']:")
  print_params(params['net'])
  # print_params(params)

  trainable_params, frozen_params = get_trainable(params, config)
  # logging.info(f'\nTrainable:\n{list(flatten_state_dict(trainable_params).keys())}'.replace(', ', ',\n')) 

  if config.optimizer == 'sgd':
    logging.info('Using SGD')
    tx = optax.sgd(
        learning_rate=learning_rate_fn,
        momentum=config.momentum,
        nesterov=True,
    )
  elif config.optimizer == 'adamw':
    logging.info(f'Using AdamW with wd {config.weight_decay}')
    tx = optax.adamw(
        learning_rate=learning_rate_fn,
        b1=config.adam_b1,
        b2=config.adam_b2,
        weight_decay=config.weight_decay,
        # mask=mask_fn,  # TODO{km}
    )
  elif config.optimizer == 'radam':
    logging.info(f'Using RAdam with wd {config.weight_decay}')
    assert config.weight_decay == 0.0
    tx = optax.radam(
        learning_rate=learning_rate_fn,
        b1=config.adam_b1,
        b2=config.adam_b2,
    )
  elif config.optimizer == 'adam':
    logging.info('Using Adam')
    tx = optax.adam(
        learning_rate=learning_rate_fn,
        b1=config.adam_b1,
        b2=config.adam_b2,
    )
  else:
    raise ValueError(f'Unknown optimizer: {config.optimizer}')

  state = TrainState(
    step=0,
    apply_fn=functools.partial(model.apply, method=model.forward),
    params=params,
    tx=tx,
    opt_state=tx.init(trainable_params),
    batch_stats=batch_stats,
    dynamic_scale=dynamic_scale,
  )
  return state


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0
  ) if not config.fid.eval_only else None

  rng = random.key(config.seed)

  image_size = config.model.image_size
  config.dataset.out_channels = config.model.out_channels

  logging.info('config.batch_size: {}'.format(config.batch_size))

  if config.batch_size % jax.process_count() > 0:
    raise ValueError('Batch size must be divisible by the number of processes')
  local_batch_size = config.batch_size // jax.process_count()
  logging.info('local_batch_size: {}'.format(local_batch_size))
  logging.info('jax.local_device_count: {}'.format(jax.local_device_count()))

  if local_batch_size % jax.local_device_count() > 0:
    raise ValueError('Local batch size must be divisible by the number of local devices')

  train_loader, steps_per_epoch = input_pipeline.create_split(
    config.dataset,
    local_batch_size,
    split='train',
    # split='val',
  )
  logging.info('steps_per_epoch: {}'.format(steps_per_epoch))

  # base_learning_rate = config.learning_rate * config.batch_size / 256.0
  logging.warning('No lr scaling.')
  base_learning_rate = config.learning_rate

  model = create_model(model_cls=models_ddpm.FMDDPM, half_precision=config.half_precision, **config.model)

  learning_rate_fn = create_learning_rate_fn(config, base_learning_rate, steps_per_epoch)

  state = create_train_state(rng, config, model, image_size, learning_rate_fn)

  if config.restore != '':
    logging.info('Restoring from: {}'.format(config.restore))
    state = restore_checkpoint(state, config.restore)
  elif config.pretrain != '':
    logging.info('Loading pre-trained from: {}'.format(config.restore))
    state = restore_pretrained(state, config.pretrain, config)

  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)
  epoch_offset = step_offset // steps_per_epoch  # sanity check for resuming
  assert epoch_offset * steps_per_epoch == step_offset
  state = jax_utils.replicate(state)

  ema_scales_fn = functools.partial(edm_ema_scales_schedules, steps_per_epoch=steps_per_epoch, config=config)
  p_train_step = jax.pmap(
    functools.partial(train_step, rng_init=rng, learning_rate_fn=learning_rate_fn, ema_scales_fn=ema_scales_fn, config=config),
    axis_name='batch',
  )
  p_sample_step = jax.pmap(
    functools.partial(sample_step, model=model, rng_init=rng, device_batch_size=config.fid.device_batch_size,),
    axis_name='batch',
  )
  vis_sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())  # for visualization
  logging_util.verbose_on()
  logging.info(f'fixed_sample_idx: {vis_sample_idx}')
  logging_util.verbose_off()

  # ------------------------------------------------------------
  logging.info('Compiling p_sample_step...')
  t_start = time.time()
  lowered = p_sample_step.lower(
    params={'params': {'net': state.params['net']}, 'batch_stats': {}},
    sample_idx=vis_sample_idx,)
  p_sample_step = lowered.compile()
  logging.info('p_sample_step compiled in {}s'.format(time.time() - t_start))

  def run_p_sample_step(p_sample_step, state, sample_idx, ema=False):
    # redefine the interface
    net_key = 'net_ema' if ema else 'net'
    images = p_sample_step(params={'params': {'net': state.params[net_key]}, 'batch_stats': {}}, sample_idx=sample_idx)
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return images[0]  # images have been all gathered

  # debugging here
  # vis = run_p_sample_step(p_sample_step, state, vis_sample_idx, ema=True)
  # vis = make_grid_visualization(vis)
  # writer.write_images(0, {'dbg': vis})
  # ------------------------------------------------------------
  
  # ------------------------------------------------------------------------------------
  if config.fid.on_use:  # we will evaluate fid    
    inception_net = fid_util.build_jax_inception()
    stats_ref = fid_util.get_reference(config.fid.cache_ref, inception_net)

    if config.fid.eval_only:
      samples_all = sample_util.generate_samples_for_fid_eval(state, workdir, config, p_sample_step, run_p_sample_step, ema=False)
      mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
      fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
      logging.info(f'w/o ema: FID at {samples_all.shape[0]} samples: {fid_score}')

      samples_all = sample_util.generate_samples_for_fid_eval(state, workdir, config, p_sample_step, run_p_sample_step, ema=True)
      mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
      fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
      logging.info(f' w/ ema: FID at {samples_all.shape[0]} samples: {fid_score}')
      return None

    # debugging here
    # samples_dir = '/kmh-nfs-us-mount/logs/kaiminghe/results-edm/edm-cifar10-32x32-uncond-vp'
    # samples = sample_util.get_samples_from_dir(samples_dir, config)
  # ------------------------------------------------------------------------------------

  train_metrics = []
  hooks = []
  # if jax.process_index() == 0:
  #   hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')
  for epoch in range(epoch_offset, config.num_epochs):
    train_loader.sampler.set_epoch(epoch)
    logging.info('epoch {}...'.format(epoch))
    for n_batch, batch in enumerate(train_loader):
      step = epoch * steps_per_epoch + n_batch
      batch = prepare_batch_data(batch, config)
      state, metrics, vis = p_train_step(state, batch)
      
      if epoch == epoch_offset and n_batch == 0:
        logging.info('p_train_step compiled in {}s'.format(time.time() - train_metrics_last_t))
        logging.info('Initial compilation completed. Reset timer.')
        train_metrics_last_t = time.time()
      
      for h in hooks:
        h(step)

      ep = step / steps_per_epoch

      if config.get('log_per_step'):
        train_metrics.append(metrics)
        if (step + 1) % config.log_per_step == 0:
          train_metrics = common_utils.get_metrics(train_metrics)
          summary = {
              f'{k}': v
              for k, v in jax.tree_util.tree_map(
                  lambda x: float(x.mean()), train_metrics
              ).items()
          }
          summary['steps_per_second'] = config.log_per_step / (time.time() - train_metrics_last_t)
          # summary['seconds_per_step'] = (time.time() - train_metrics_last_t) / config.log_per_step

          # step for tensorboard
          summary["ep"] = ep

          writer.write_scalars(step + 1, summary)
          train_metrics = []
          train_metrics_last_t = time.time()

    # logging
    if (epoch + 1) % config.visualize_per_epoch == 0:
      vis = visualize_cifar_batch(vis)
      writer.write_images(epoch + 1, {'vis_train': vis})

    # Show samples (eval)
    if (epoch + 1) % config.eval_per_epoch == 0:
      logging.info('Sample epoch {}...'.format(epoch))
      # ------------------------------------------------------------
      vis = run_p_sample_step(p_sample_step, state, vis_sample_idx, ema=False)
      vis = make_grid_visualization(vis)
      # vis_ema = run_p_sample_step(p_sample_step, state, vis_sample_idx, ema=True)
      # vis_ema = make_grid_visualization(vis_ema)
      # sep = jnp.zeros_like(vis)[:, :8,]  # separator
      # vis = jnp.concatenate([vis, sep, vis_ema], axis=1)
      writer.write_images(epoch + 1, {'vis_sample': vis})
      # ------------------------------------------------------------
      writer.flush()

    # save checkpoint
    if (
      (epoch + 1) % config.checkpoint_per_epoch == 0
      or epoch == config.num_epochs
      or epoch == 0  # saving at the first epoch for sanity check
    ):
      state = sync_batch_stats(state)
      # TODO{km}: suppress the annoying warning.
      save_checkpoint(state, workdir)
      logging.info(f'Work dir: {workdir}')  # for monitoring

    if config.fid.on_use and (
      (epoch + 1) % config.fid.fid_per_epoch == 0
      or epoch == config.num_epochs
      # or epoch == 0
    ):
      samples_all = sample_util.generate_samples_for_fid_eval(state, workdir, config, p_sample_step, run_p_sample_step, ema=False)
      mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
      fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
      logging.info(f'w/o ema: FID at {samples_all.shape[0]} samples: {fid_score}')
      writer.write_scalars(epoch + 1, {'FID': fid_score})
      writer.flush()

      # ema results are much better
      samples_all = sample_util.generate_samples_for_fid_eval(state, workdir, config, p_sample_step, run_p_sample_step, ema=True)
      mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
      fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
      logging.info(f' w/ ema: FID at {samples_all.shape[0]} samples: {fid_score}')
      writer.write_scalars(epoch + 1, {'FID_ema': fid_score})
      writer.flush()

      vis = make_grid_visualization(samples_all, to_uint8=False)
      writer.write_images(epoch + 1, {'vis_sample_ema': vis})

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.key(0), ()).block_until_ready()

  return state
