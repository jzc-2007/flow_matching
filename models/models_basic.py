from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


ModuleDef = Any


class TwoLayerMLP(nn.Module):
  # Define the size of each layer
  hidden_dim: int
  output_dim: int

  @nn.compact
  def __call__(self, x):

    # First layer
    x = nn.Dense(self.hidden_dim)(x)
    x = nn.relu(x)

    # Second layer
    x = nn.Dense(self.output_dim)(x)

    return x


class FlaxDiagonalGaussianDistribution(object):
  def __init__(self, parameters, has_logvar=True, deterministic=False):
    # Last axis to account for channels-last
    if has_logvar:
      self.mean, self.logvar = jnp.split(parameters, 2, axis=-1)
    else:
      self.mean = parameters
      self.logvar = 0.0
    self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
    self.deterministic = deterministic
    self.std = jnp.exp(0.5 * self.logvar)
    self.var = jnp.exp(self.logvar)
    if self.deterministic:
      self.var = self.std = jnp.zeros_like(self.mean)

  def sample(self, key):
    return self.mean + self.std * jax.random.normal(key, self.mean.shape)

  def kl(self, other=None):
    if self.deterministic:
      return jnp.array([0.0])

    if other is None:
      return 0.5 * jnp.sum(self.mean**2 + self.var - 1.0 - self.logvar, axis=[1, 2, 3])

    return 0.5 * jnp.sum(
      jnp.square(self.mean - other.mean) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
      axis=[1, 2, 3],
    )

  def nll(self, sample, axis=[1, 2, 3]):
    if self.deterministic:
      return jnp.array([0.0])

    logtwopi = jnp.log(2.0 * jnp.pi)
    return 0.5 * jnp.sum(logtwopi + self.logvar + jnp.square(sample - self.mean) / self.var, axis=axis)

  def mode(self):
    return self.mean

  def loss_kl(self):
    kldiv = self.kl()
    _, h, w, c = self.mean.shape
    loss_kl = kldiv / (h * w * c)  # mean over spatial and channel dims
    return loss_kl


class FlaxDownsample2D(nn.Module):
  """
  Flax implementation of 2D Downsample layer

  Args:
    out_channels (`int`):
      Output channels
    dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
      Parameters `dtype`
  """

  out_channels: int
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.conv = nn.Conv(
      self.out_channels,
      kernel_size=(3, 3),
      strides=(2, 2),
      padding="VALID",
      dtype=self.dtype,
    )

  def __call__(self, hidden_states):
    pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
    hidden_states = jnp.pad(hidden_states, pad_width=pad)
    hidden_states = self.conv(hidden_states)
    return hidden_states


class FlaxUpsample2D(nn.Module):
  """
  Flax implementation of 2D Upsample layer

  Args:
    out_channels (`int`):
      Input channels
    dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
      Parameters `dtype`
  """

  out_channels: int
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.conv = nn.Conv(
      self.out_channels,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding=((1, 1), (1, 1)),
      dtype=self.dtype,
    )

  def __call__(self, hidden_states):
    batch, height, width, channels = hidden_states.shape
    hidden_states = jax.image.resize(
      hidden_states,
      shape=(batch, height * 2, width * 2, channels),
      method="nearest",
    )
    hidden_states = self.conv(hidden_states)
    return hidden_states


class FlaxResnetBlock2D(nn.Module):
  """
  Flax implementation of 2D Resnet Block.

  Args:
    in_channels (`int`):
      Input channels
    out_channels (`int`):
      Output channels
    dropout (:obj:`float`, *optional*, defaults to 0.0):
      Dropout rate
    groups (:obj:`int`, *optional*, defaults to `32`):
      The number of groups to use for group norm.
    use_nin_shortcut (:obj:`bool`, *optional*, defaults to `None`):
      Whether to use `nin_shortcut`. This activates a new layer inside ResNet block
    dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
      Parameters `dtype`
  """

  in_channels: int
  out_channels: int = None
  groups: int = 16
  use_nin_shortcut: bool = None
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    out_channels = self.in_channels if self.out_channels is None else self.out_channels

    self.norm1 = nn.GroupNorm(num_groups=self.groups, epsilon=1e-6)
    self.conv1 = nn.Conv(
      out_channels,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding=((1, 1), (1, 1)),
      dtype=self.dtype,
    )

    self.norm2 = nn.GroupNorm(num_groups=self.groups, epsilon=1e-6)
    self.conv2 = nn.Conv(
      out_channels,
      kernel_size=(3, 3),
      strides=(1, 1),
      padding=((1, 1), (1, 1)),
      dtype=self.dtype,
    )

    use_nin_shortcut = self.in_channels != out_channels if self.use_nin_shortcut is None else self.use_nin_shortcut

    self.conv_shortcut = None
    if use_nin_shortcut:
      raise NotImplementedError
      self.conv_shortcut = nn.Conv(
        out_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="VALID",
        dtype=self.dtype,
      )

  def __call__(self, hidden_states):
    residual = hidden_states
    hidden_states = self.norm1(hidden_states)
    hidden_states = nn.swish(hidden_states)
    hidden_states = self.conv1(hidden_states)

    hidden_states = self.norm2(hidden_states)
    hidden_states = nn.swish(hidden_states)
    hidden_states = self.conv2(hidden_states)

    if self.conv_shortcut is not None:
      residual = self.conv_shortcut(residual)

    return hidden_states + residual