from typing import Any
import jax
import optax
from flax.training import train_state
from flax.training import dynamic_scale as dynamic_scale_lib


def write_back_params(original_params, updated_params):
    """ Recursively merge updated_params back into original_params. """
    def update(orig, updates):
        if isinstance(orig, dict):
            for key, val in updates.items():
                if key in orig:
                    orig[key] = update(orig[key], val)
        else:
            orig = updates
        return orig

    return update(original_params, updated_params)


class TrainState(train_state.TrainState):
  batch_stats: Any
  dynamic_scale: dynamic_scale_lib.DynamicScale

  def apply_gradients(self, *, grads, params, **kwargs):
    grads_with_opt = grads
    params_with_opt = params

    updates, new_opt_state = self.tx.update(
      grads_with_opt, self.opt_state, params_with_opt
    )
    new_params_with_opt = optax.apply_updates(params_with_opt, updates)

    new_params = write_back_params(self.params, new_params_with_opt)
    assert jax.tree_structure(new_params) == jax.tree_structure(self.params)

    return self.replace(
      step=self.step + 1,
      params=new_params,
      opt_state=new_opt_state,
      **kwargs,
    )
