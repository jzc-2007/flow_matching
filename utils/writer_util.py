# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import tensorflow as tf

from clu.metric_writers import SummaryWriter


def write_scalars(self, step: int, scalars):
    """Revise write_scalars to support epoch_1000x
    
    Kaiming: When changing batch sizes, it is more informative
    to compare different runs in epochs, not just iterations.
    However, tensorboard doesn't allow x-axis to be float. So we
    use epoch_1000x, which is int, to address this issue.
    """
    with self._summary_writer.as_default():
        # only log the ep-based scalars
        if "ep" in scalars:
            ep_1000x = int(scalars["ep"] * 1000)  # 1/1000 ep as a unit
            for key, value in scalars.items():
                if key != 'ep':
                    tf.summary.scalar('ep_' + key, value, step=ep_1000x)
                else:
                    tf.summary.scalar('ep', value, step=ep_1000x)
        else:
            for key, value in scalars.items():
                tf.summary.scalar(key, value, step=step)

SummaryWriter.write_scalars = write_scalars