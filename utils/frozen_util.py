import jax
from absl import logging

from utils.state_utils import flatten_state_dict
from flax import traverse_util


def merge_params(params_a, params_b):
  merged = {**flatten_state_dict(params_a), **flatten_state_dict(params_b)}
  return traverse_util.unflatten_dict(merged, sep="/")


def extract_trainable_parameters(params, trainable_prefixes):

  flat_params = flatten_state_dict(params)
  
  trainable_params = {}
  for k, v in flat_params.items():
    for prefix in trainable_prefixes:
      if k.startswith(prefix):
        trainable_params[k] = v
        break

  for k in trainable_params.keys():
    flat_params.pop(k)

  # logging.info(f'trainable: \n{trainable_params.keys()}'.replace(', ', ',\n'))

  trainable_params_dict = traverse_util.unflatten_dict(trainable_params, sep="/")
  frozen_params_dict = traverse_util.unflatten_dict(flat_params, sep="/")

  # sanity check:
  merged = merge_params(trainable_params_dict, frozen_params_dict)
  assert jax.tree_structure(merged) == jax.tree_structure(params)

  return trainable_params_dict, frozen_params_dict