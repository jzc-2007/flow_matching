import torch
from torch import Tensor


"""
Copied from torchscript_inception
"""

def forward(
    img: Tensor,
    # return_features: bool=False,
    # use_fp16: bool=False,
    # no_output_bias: bool=False
    ) -> Tensor:

  _0 = torch.nn.functional.affine_grid
  _1 = torch.nn.functional.grid_sample
  batch_size, channels, height, width, = img.shape
  # if torch.eq(channels, 3):
  #   pass
  # else:
  #   ops.prim.RaiseException("AssertionError: ")
  # if use_fp16:
  #   _2 = 5
  # else:
  #   _2 = 6
  # x = torch.to(img, _2)
  x = img
  # theta = torch.eye(2, 3, dtype=None, layout=None, device=ops.prim.device(x))
  theta = torch.eye(2, 3, dtype=None, layout=None)
  _3 = torch.select(torch.select(theta, 0, 0), 0, 2)
  _4 = torch.select(torch.select(theta, 0, 0), 0, 0)
  _5 = torch.div(_4, width)
  _6 = torch.select(torch.select(theta, 0, 0), 0, 0)
  # _7 = torch.add_(_3, torch.sub(_5, torch.div(_6, 299)))
  _7 = torch.add(_3, torch.sub(_5, torch.div(_6, 299)))
  _8 = torch.select(torch.select(theta, 0, 1), 0, 2)
  _9 = torch.select(torch.select(theta, 0, 1), 0, 1)
  _10 = torch.div(_9, height)
  _11 = torch.select(torch.select(theta, 0, 1), 0, 1)
  # _12 = torch.add_(_8, torch.sub(_10, torch.div(_11, 299)))
  _12 = torch.add(_8, torch.sub(_10, torch.div(_11, 299)))
  # _13 = torch.unsqueeze(torch.to(theta, ops.prim.dtype(x)), 0)
  _13 = torch.unsqueeze(theta, 0)
  # theta0 = torch.repeat(_13, [batch_size, 1, 1])
  theta0 = _13.repeat([batch_size, 1, 1])
  grid = _0(theta0, [batch_size, channels, 299, 299], False, )
  x0 = _1(x, grid, "bilinear", "border", False, )
  x1 = torch.sub(x0, 128)
  x2 = torch.div(x1, 128)
  # layers = self.layers
  # _14 = torch.reshape((layers).forward(x2, ), [-1, 2048])
  # features = torch.to(_14, 6)
  return x2
