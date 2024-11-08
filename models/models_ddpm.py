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

"""Flax implementation of ResNet V1.5."""

# See issue #620.
# pytype: disable=wrong-arg-count

from absl import logging
from typing import Any, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from functools import partial

from models.models_unet import ContextUnet
from models.models_ncsnpp import NCSNppFM
import models.jcm.sde_lib as sde_lib
from models.jcm.sde_lib import batch_mul



ModuleDef = Any


def ct_ema_scales_schedules(step, config, steps_per_epoch):
  start_ema = float(config.ct.start_ema)
  start_scales = int(config.ct.start_scales)
  end_scales = int(config.ct.end_scales)
  total_steps = config.num_epochs * steps_per_epoch

  scales = jnp.ceil(jnp.sqrt((step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2) + start_scales**2) - 1).astype(jnp.int32)
  scales = jnp.maximum(scales, 1)
  c = -jnp.log(start_ema) * start_scales
  target_ema = jnp.exp(-c / scales)
  scales = scales + 1
  return target_ema, scales


def edm_ema_scales_schedules(step, config, steps_per_epoch):
  ema_halflife_kimg = 500  # from edm
  ema_halflife_nimg = ema_halflife_kimg * 1000

  ema_rampup_ratio = 0.05
  ema_halflife_nimg = jnp.minimum(ema_halflife_nimg, step * config.batch_size * ema_rampup_ratio)

  ema_beta = 0.5 ** (config.batch_size / jnp.maximum(ema_halflife_nimg, 1e-8))
  scales = jnp.ones((1,), dtype=jnp.int32)
  return ema_beta, scales


# move this out from model for JAX compilation
def generate(params, model, rng, n_sample):
  """
  Generate samples from the model
  """

  # prepare schedule
  num_steps = model.n_T
  step_indices = jnp.arange(num_steps, dtype=model.dtype)
  t_steps = step_indices / num_steps  # t_i = i / N
  t_steps = jnp.concatenate([t_steps, jnp.ones((1,), dtype=model.dtype)], axis=0)  # t_N = 1; no need to round_sigma

  # initialize noise
  x_shape = (n_sample, model.image_size, model.image_size, model.out_channels)
  rng_used, rng = jax.random.split(rng, 2)
  x_0 = jax.random.normal(rng_used, x_shape, dtype=model.dtype)

  def step_fn(i, inputs):
    x_i = inputs
    x_i, _ = model.apply(
        params,  # which is {'params': state.params, 'batch_stats': state.batch_stats},
        x_i, i, t_steps,
        method=model.sample_one_step,
        mutable=['batch_stats'],
    )
    outputs = x_i
    return outputs

  outputs = jax.lax.fori_loop(0, num_steps, step_fn, x_0)
  images = outputs
  return images


class FMDDPM(nn.Module):
  """DDPM for flow matching"""
  image_size: int
  base_width: int
  num_classes: int = 10
  out_channels: int = 1
  P_std: float = 1.2
  P_mean: float = -1.2
  n_T: int = 18  # inference steps
  net_type: str = 'ncsnpp'
  dropout: float = 0.15
  dtype: Any = jnp.float32
  use_aug_label: bool = False

  def setup(self):
    net_fn = partial(NCSNppFM,
      base_width=self.base_width,
      image_size=self.image_size,
      dropout=self.dropout)
    self.net = net_fn(name='net')
    self.net_ema = net_fn(name='net_ema')


  def get_visualization(self, list_imgs):
    vis = jnp.concatenate(list_imgs, axis=1)
    return vis
    
  # def compute_losses(self, pred, gt):
  #   assert pred.shape == gt.shape
  #   loss_rec = jnp.mean((pred - gt)**2)
  #   loss_train = loss_rec
  #   dict_losses = {
  #     'loss_rec': loss_rec,
  #     'loss_train': loss_train
  #   }
  #   return loss_train, dict_losses

  def sample_one_step(self, x_i, i, t_steps):
    x_cur = x_i
    t_cur, t_next = t_steps[i], t_steps[i + 1]
    t_cur = jnp.repeat(t_cur, x_cur.shape[0])
    t_next = jnp.repeat(t_next, x_cur.shape[0])
    velocity = self.net(x_cur, t_cur, train=False)
    x_next = x_cur + batch_mul(velocity, (t_next - t_cur).reshape(-1, 1, 1, 1))
    return x_next

  def forward(self, imgs, train: bool = True):
    imgs = imgs.astype(self.dtype)
    x = imgs
    bz = imgs.shape[0]

    timesteps = jax.random.uniform(self.make_rng('gen'), [bz, 1, 1, 1], minval=0.0, maxval=1.0)
    noise = jax.random.normal(self.make_rng('gen'), x.shape, dtype=self.dtype)
    x_t = (1 - timesteps) * noise + timesteps * x
    velocity = x - noise
    pred = self.net(x_t, timesteps.reshape(bz), train=train)

    loss = (pred - velocity)**2
    loss = jnp.sum(loss, axis=(1, 2, 3))  # sum over pixels
    loss = loss.mean()  # mean over batch
    
    loss_train = loss

    dict_losses = {}
    dict_losses['loss'] = loss  # legacy
    dict_losses['loss_train'] = loss_train

    images = self.get_visualization([imgs, x_t, x_t + pred * (1 - timesteps)])
    return loss_train, dict_losses, images

  def __call__(self, imgs, train: bool = False):
    t = jnp.ones((imgs.shape[0],))
    out = self.net(x=imgs, time_cond=t)
    out_ema = None   # no need to initialize it here
    return out, out_ema
  



# class SimDDPM(nn.Module):
#   """Simple DDPM."""
#   image_size: int
#   base_width: int
#   num_classes: int = 10
#   out_channels: int = 1
#   P_std: float = 1.2
#   P_mean: float = -1.2
#   n_T: int = 18  # inference steps
#   net_type: str = 'ncsnpp'
#   dropout: float = 0.0
#   dtype: Any = jnp.float32
#   use_aug_label: bool = False

#   def setup(self):

#     sde = sde_lib.KVESDE(
#       t_min=0.002,
#       t_max=80.0,
#       N=18,  # config.model.num_scales
#       rho=7.0,
#       data_std=0.5,
#     )
#     self.sde = sde

#     if self.net_type == 'context':
#       net_fn = partial(ContextUnet,
#         in_channels=self.out_channels,
#         n_feat=self.base_width,
#         n_classes=self.num_classes,
#         image_size=self.image_size,)
#     elif self.net_type == 'ncsnpp':
#       net_fn = partial(NCSNpp,
#         base_width=self.base_width,
#         image_size=self.image_size,
#         dropout=self.dropout)
#     elif self.net_type == 'ncsnppedm':
#       net_fn = partial(NCSNppEDM,
#         base_width=self.base_width,
#         image_size=self.image_size,
#         dropout=self.dropout)
#     else:
#       raise ValueError(f'Unknown net type: {self.net_type}')

#     # declare two networks
#     self.net = net_fn(name='net')
#     self.net_ema = net_fn(name='net_ema')


#   def get_visualization(self, list_imgs):
#     vis = jnp.concatenate(list_imgs, axis=1)
#     return vis
    
#   def compute_losses(self, pred, gt):
#     assert pred.shape == gt.shape

#     # simple l2 loss
#     loss_rec = jnp.mean((pred - gt)**2)
    
#     loss_train = loss_rec

#     dict_losses = {
#       'loss_rec': loss_rec,
#       'loss_train': loss_train
#     }
#     return loss_train, dict_losses

#   def sample_one_step(self, x_i, rng, i, t_steps):

#     x_cur = x_i
#     t_cur = t_steps[i]
#     t_next = t_steps[i + 1]

#     # TODO{kaiming}: revisit S_churn
#     t_hat = t_cur
#     x_hat = x_cur  # x_hat is always x_cur when gamma=0

#     t_hat = jnp.repeat(t_hat, x_hat.shape[0])
#     t_next = jnp.repeat(t_next, x_hat.shape[0])
    
#     # TODO: ema net?
#     # Euler step.
#     denoised = self.forward_edm_denoising_function(x_hat, t_hat, train=False)
#     d_cur = batch_mul(x_hat - denoised, 1. / t_hat)
#     x_next = x_hat + batch_mul(d_cur, t_next - t_hat)

#     # Apply 2nd order correction
#     denoised = self.forward_edm_denoising_function(x_next, t_next, train=False)
#     d_prime = batch_mul(x_next - denoised, 1. / jnp.maximum(t_next, 1e-8))  # won't take effect if t_next is 0 (last step)
#     x_next_ = x_hat + batch_mul(0.5 * d_cur + 0.5 * d_prime, t_next - t_hat)

#     x_next = jnp.where(i < self.n_T - 1, x_next_, x_next)

#     return x_next

#   def compute_t(self, indices, scales):
#     sde = self.sde
#     t = sde.t_max ** (1 / sde.rho) + indices / (scales - 1) * (
#         sde.t_min ** (1 / sde.rho) - sde.t_max ** (1 / sde.rho)
#     )
#     t = t**sde.rho
#     return t

#   def forward_edm_denoising_function(self, x, sigma, augment_label=None, train: bool = True):  # EDM
#     c_skip = self.sde.data_std ** 2 / (sigma ** 2 + self.sde.data_std ** 2)
#     c_out = sigma * self.sde.data_std / jnp.sqrt(sigma ** 2 + self.sde.data_std ** 2)

#     c_in = 1 / jnp.sqrt(sigma ** 2 + self.sde.data_std ** 2)
#     c_noise = 0.25 * jnp.log(sigma)

#     # forward network
#     in_x = batch_mul(x, c_in)
#     c_noise = c_noise.reshape(c_noise.shape[0])

#     F_x = self.net(in_x, c_noise, augment_label=augment_label, train=train)

#     D_x = batch_mul(x, c_skip) + batch_mul(F_x, c_out)
#     return D_x

#   def forward(self, imgs, labels, augment_label, train: bool = True):
#     imgs = imgs.astype(self.dtype)
#     gt = imgs
#     x = imgs
#     bz = imgs.shape[0]

#     # -----------------------------------------------------------------

#     rnd_normal = jax.random.normal(self.make_rng('gen'), [bz, 1, 1, 1], dtype=self.dtype)
#     sigma = jnp.exp(rnd_normal * self.P_std + self.P_mean)

#     weight = (sigma ** 2 + self.sde.data_std ** 2) / (sigma * self.sde.data_std) ** 2

#     noise = jax.random.normal(self.make_rng('gen'), x.shape, dtype=self.dtype) * sigma

#     xn = x + noise
#     D_xn = self.forward_edm_denoising_function(xn, sigma, augment_label)

#     loss = (D_xn - gt)**2
#     loss = weight * loss
#     loss = jnp.sum(loss, axis=(1, 2, 3))  # sum over pixels
#     loss = loss.mean()  # mean over batch
    
#     loss_train = loss

#     dict_losses = {}
#     dict_losses['loss'] = loss  # legacy
#     dict_losses['loss_train'] = loss_train

#     images = self.get_visualization([gt, xn, D_xn])
#     return loss_train, dict_losses, images

#   def __call__(self, imgs, labels, train: bool = False):
#     # initialization only
#     t = jnp.ones((imgs.shape[0],))
#     augment_label = jnp.ones((imgs.shape[0], 9)) if self.use_aug_label else None  # fixed augment_dim
#     out = self.net(imgs, t, augment_label)
#     out_ema = None   # no need to initialize it here
#     return out, out_ema
