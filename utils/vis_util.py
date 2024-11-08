import jax
import jax.numpy as jnp
import numpy as np


def float_to_uint8(x):
  x = x * 0.5 + 0.5
  x = jnp.clip(x * 255, 0, 255).astype(jnp.uint8)
  x = jax.device_get(x)
  return x


def visualize_cifar_batch(vis, col=8, max_bz=64):
  assert vis.ndim == 5
  m, n, h, w, c = vis.shape
  vis = vis.reshape((-1, h, w, c))
  bz = min(vis.shape[0], max_bz)  # only visualize 64
  vis = vis[:bz]

  n, h, w, c = vis.shape
  col = min(col, n)  # number of columns
  vis = vis.reshape((-1, col, h, w, c))
  vis = jnp.einsum('nlhwc->nhlwc', vis).reshape((-1, h, col * w, c))
  # vis = jnp.clip(vis * 0.5 + 0.5, 0, 1)
  vis = float_to_uint8(vis)
  return vis


def make_grid_visualization(vis, grid=8, max_bz=8, to_uint8=True):
  assert vis.ndim == 4, vis.shape
  n, h, w, c = vis.shape

  col = grid
  row = min(grid, n // col) 
  if n % (col * row) != 0:
    n = col * row * max_bz
    vis = vis[:n]
    n, h, w, c = vis.shape
  assert n % (col * row) == 0

  vis = vis.reshape((-1, col, row * h, w, c))
  vis = jnp.einsum('mlhwc->mhlwc', vis)
  vis = vis.reshape((-1, row * h, col * w, c))

  bz = min(vis.shape[0], max_bz)  # only visualize 64
  vis = vis[:bz]

  # vis = jnp.clip(vis * 0.5 + 0.5, 0, 1)
  if to_uint8:
    vis = float_to_uint8(vis)
  return vis
