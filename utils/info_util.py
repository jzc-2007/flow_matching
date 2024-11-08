from absl import logging
import utils.state_utils as state_utils


# Function to print number of parameters
def print_params(params):
    params_flatten = state_utils.flatten_state_dict(params)

    total_params = 0
    max_length = max(len(k) for k in params_flatten.keys())
    max_shape = max(len(f"{p.shape}") for p in params_flatten.values())
    max_digits = max(len(f"{p.size:,}") for p in params_flatten.values())
    logging.info('-' * (max_length + max_digits + max_shape + 8))
    for name, param in params_flatten.items():
        layer_params = param.size
        str_layer_shape = f"{param.shape}".rjust(max_shape) 
        str_layer_params = f"{layer_params:,}".rjust(max_digits)
        logging.info(f" {name.ljust(max_length)} | {str_layer_shape} | {str_layer_params} ")
        total_params += layer_params
    logging.info('-' * (max_length + max_digits + max_shape + 8))
    logging.info(f"Total parameters: {total_params:,}")