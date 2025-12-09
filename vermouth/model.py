"""
@Author: Conghao Wong
@Date: 2025-12-02 11:10:53
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-02 16:06:45
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model
from qpid.training import Structure


class VermouthModel(Model):
    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        obs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        return torch.repeat_interleave(
            obs[..., -1:, :],
            repeats=self.args.pred_frames,
            dim=-2)
    

class Vermouth(Structure):
    MODEL_TYPE = VermouthModel
    is_trainable = False
