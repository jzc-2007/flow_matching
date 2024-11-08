from absl import logging
from flax import linen as nn
import jax.numpy as jnp
import jax


class Sequential(nn.Module):
  layers: list

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


class ResidualConvBlock(nn.Module):
  in_channels: int
  out_channels: int
  groups: int = 4
  is_res: bool = False

  @nn.compact
  def __call__(self, x):
    input = x  # keep for residual connection

    x = nn.Conv(self.out_channels, (3, 3), padding='SAME', name='conv1')(x)
    x = nn.GroupNorm(num_groups=self.groups, epsilon=1e-6, name='gn1')(x)
    x = nn.gelu(x)
    conv1 = x  # backup

    x = nn.Conv(self.out_channels, (3, 3), padding='SAME', name='conv2')(x)
    x = nn.GroupNorm(num_groups=self.groups, epsilon=1e-6, name='gn2')(x)
    x = nn.gelu(x)

    if self.is_res:
      if self.in_channels == self.out_channels:
        out = input + x
      else:
        out = conv1 + x
      return out / jnp.sqrt(2)  # why?
    else:
      return x


class UnetDown(nn.Module):
  in_channels: int
  out_channels: int

  @nn.compact
  def __call__(self, x):
    x = ResidualConvBlock(self.in_channels, self.out_channels)(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')
    return x


class UnetUp(nn.Module):
  in_channels: int
  out_channels: int

  @nn.compact
  def __call__(self, x, skip):
    x = jnp.concatenate((x, skip), axis=-1)
    x = nn.ConvTranspose(self.out_channels, kernel_size=(2, 2), strides=(2, 2), padding='SAME', name='upconv')(x)
    x = ResidualConvBlock(self.out_channels, self.out_channels)(x)
    x = ResidualConvBlock(self.out_channels, self.out_channels)(x)
    return x


class EmbedFC(nn.Module):
  input_dim: int
  emb_dim: int

  @nn.compact
  def __call__(self, x):
    x = x.reshape(-1, self.input_dim)  # Reshape input to match dimensions
    x = nn.Dense(self.emb_dim)(x)
    x = nn.gelu(x)
    x = nn.Dense(self.emb_dim)(x)
    return x


class ContextUnet(nn.Module):
  in_channels: int
  image_size: int
  n_feat: int = 256
  n_classes: int = 10

  def setup(self):
    self.init_conv = ResidualConvBlock(self.in_channels, self.n_feat, is_res=True)
    self.down1 = UnetDown(self.n_feat, self.n_feat)
    self.down2 = UnetDown(self.n_feat, 2 * self.n_feat)

    self.timeembed1 = EmbedFC(1, 2*self.n_feat)
    self.timeembed2 = EmbedFC(1, 1*self.n_feat)

    s = self.image_size // 2**2
    self.up0 = Sequential(layers=[
      nn.ConvTranspose(2 * self.n_feat, kernel_size=(s, s), strides=(s, s), padding='SAME', name='upconv'),
      nn.GroupNorm(),
      nn.relu,]
    )

    self.up1 = UnetUp(4 * self.n_feat, self.n_feat)
    self.up2 = UnetUp(2 * self.n_feat, self.n_feat)

    self.out = Sequential(layers=[
      nn.Conv(self.n_feat, kernel_size=(3, 3), strides=(1, 1), padding='SAME',),
      nn.GroupNorm(),
      nn.relu,
      nn.Conv(self.in_channels, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name='conv_pix'),]
    )

  def __call__(self, imgs, t, verbose=True):

    x = imgs

    logging_fn = logging.info if verbose else lambda x: None

    x = self.init_conv(x)
    logging_fn(f'x shape: {x.shape}')
    down1 = self.down1(x)
    logging_fn(f'down1 shape: {down1.shape}')
    down2 = self.down2(down1)
    logging_fn(f'down2 shape: {down2.shape}')
    hiddenvec = nn.avg_pool(down2, (7, 7), strides=(7, 7))
    logging_fn(f'hiddenvec shape: {hiddenvec.shape}')

    cemb1 = 1
    cemb2 = 1

    temb1 = self.timeembed1(t).reshape(-1, 1, 1, self.n_feat * 2)
    temb2 = self.timeembed2(t).reshape(-1, 1, 1, self.n_feat)

    up1 = self.up0(hiddenvec)
    logging_fn(f'up1 shape: {up1.shape}')
    up2 = self.up1(cemb1 * up1 + temb1, down2)
    logging_fn(f'up2 shape: {up2.shape}')
    up3 = self.up2(cemb2 * up2 + temb2, down1)
    logging_fn(f'up3 shape: {up3.shape}')

    out = jnp.concatenate((up3, x), axis=-1)
    out = self.out(out)
    logging_fn(f'out shape: {out.shape}')
    return out


