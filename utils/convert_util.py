from absl import logging
import jax
from flax.training import checkpoints
import numpy as np
from utils.state_utils import flatten_state_dict
from flax import traverse_util

from termcolor import colored

def load_from_pytorch(params):
  """
  Kaiming: this code will convert the EDM Pytorch checkpoint to JAX checkpoint.
  As the EDM checkpoint was a persistent class object, we need to load it Pytorch and dump its state_dict before using this function.

  Usage: add load_from_pytorch(params) after initializing the model in the JAX code.
  """
  ckpt_path = '/kmh-nfs-us-mount/logs/kaiminghe/edm/240819-205048-0v5c8-he-l40s-1-g4-b512-uncond/dict-network-snapshot-025088.pkl'
  import torch
  import pickle
  with open(ckpt_path, 'rb') as f:
    net = pickle.load(f)

  flat_params = flatten_state_dict(params)

  # split attention
  net_attn = {}
  for pt_key in net.keys():
    if 'qkv' in pt_key:
      weights = net[pt_key]
      # [768, 256, 1, 1] => [256, 3, 256, 1, 1]
      weights = weights.reshape(-1, 3, *weights.shape[1:])
      
      net_attn[pt_key.replace('qkv', 'q')] = weights[:, 0]
      net_attn[pt_key.replace('qkv', 'k')] = weights[:, 1]
      net_attn[pt_key.replace('qkv', 'v')] = weights[:, 2]
    else:
      net_attn[pt_key] = net[pt_key]

  key_maps = {}
  for pt_key in net_attn:

    jx_key = pt_key
    jx_key = pt_key.replace('model.', 'net.')
    jx_key = jx_key.replace('enc.', 'enc_')
    jx_key = jx_key.replace('dec.', 'dec_')

    jx_key = jx_key.replace('norm.weight', 'norm.scale')
    jx_key = jx_key.replace('norm0.weight', 'norm0.scale')
    jx_key = jx_key.replace('norm1.weight', 'norm1.scale')

    jx_key = jx_key.replace('norm0.', 'GroupNorm_0.')
    jx_key = jx_key.replace('norm1.', 'GroupNorm_1.')

    jx_key = jx_key.replace('conv0.', 'Conv_0.')
    jx_key = jx_key.replace('conv1.', 'Conv_1.')

    jx_key = jx_key.replace('Conv_0.weight', 'Conv_0.kernel')
    jx_key = jx_key.replace('Conv_1.weight', 'Conv_1.kernel')

    jx_key = jx_key.replace('conv.weight', 'conv.kernel')

    jx_key = jx_key.replace('skip.weight', 'Conv_2.kernel')
    jx_key = jx_key.replace('skip.bias', 'Conv_2.bias')

    jx_key = jx_key.replace('affine.weight', 'Dense_0.kernel')
    jx_key = jx_key.replace('affine.bias', 'Dense_0.bias')

    # attention: NIN0: q, NIN1: k, NIN2: v
    jx_key = jx_key.replace('.q.weight', '_attn.NIN_0.W')
    jx_key = jx_key.replace('.k.weight', '_attn.NIN_1.W')
    jx_key = jx_key.replace('.v.weight', '_attn.NIN_2.W')

    jx_key = jx_key.replace('.q.bias', '_attn.NIN_0.b')
    jx_key = jx_key.replace('.k.bias', '_attn.NIN_1.b')
    jx_key = jx_key.replace('.v.bias', '_attn.NIN_2.b')

    jx_key = jx_key.replace('.proj.weight', '_attn.NIN_3.W')
    jx_key = jx_key.replace('.proj.bias', '_attn.NIN_3.b')

    jx_key = jx_key.replace('.norm2.weight', '_attn.GroupNorm_0.scale')
    jx_key = jx_key.replace('.norm2.bias', '_attn.GroupNorm_0.bias')

    jx_key = jx_key.replace('aux_residual.weight', 'aux_residual.Conv2d_0.weight')
    jx_key = jx_key.replace('aux_residual.bias', 'aux_residual.Conv2d_0.bias')

    jx_key = jx_key.replace('map_layer0.weight', 'map_layer0.kernel')
    jx_key = jx_key.replace('map_layer1.weight', 'map_layer1.kernel')

    jx_key = jx_key.replace('.', '/')
    if jx_key not in flat_params:
      logging.info(colored(f'{pt_key:48s} -> {jx_key}', 'red'))
      # print(net_attn[pt_key])
    else:
      key_maps[jx_key] = pt_key
      # print(colored(f'{pt_key:48s} -> {jx_key}', 'green'))
      pass

  # sanity
  for k in flat_params.keys():
    assert k in key_maps, f'{k} not in key_maps'

  converted_flat_params = {}
  for jx_key, pt_key in key_maps.items():
    
    pt_param = net_attn[pt_key]
    pt_shape_old = list(pt_param.shape)
    jx_shape = list(flat_params[jx_key].shape)

    pt_param = pt_param.detach().numpy()
    if pt_param.ndim == 4:
      pt_param = np.einsum('oihw->hwio', pt_param)
      if pt_param.shape[0] == pt_param.shape[1] == 1 and len(jx_shape) == 2:
        pt_param = pt_param.squeeze(0).squeeze(0)
    elif pt_param.ndim == 2:
      pt_param = np.einsum('oi->io', pt_param)

    if list(pt_param.shape) != jx_shape:
      print(colored(f'{pt_key:48s} -> {jx_key:48s} : {str(pt_shape_old):32s} => {jx_shape}', 'red'))
      raise NotImplementedError
    else:
      # print(colored(f'{pt_key:48s} -> {jx_key:48s} : {str(pt_shape_old):32s} => {jx_shape}', 'green'))
      converted_flat_params[jx_key] = jax.numpy.array(pt_param)

  converted_params = traverse_util.unflatten_dict(converted_flat_params, sep="/")

  # sanity check:
  assert jax.tree_structure(converted_params) == jax.tree_structure(params)
  return converted_params


    
  