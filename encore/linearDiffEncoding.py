"""
@Author: Conghao Wong
@Date: 2025-12-23 10:17:06
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-23 15:29:26
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers
from qpid.model.layers.transfroms import _BaseTransformLayer


class LinearDiffEncoding(torch.nn.Module):
    """
    Linear Difference Encoding Layer
    ---
    It is used to encode the difference between the observed trajectory and
    the corresponding linear (least square) trajectory of the ego agent.
    """

    def __init__(self, obs_frames: int,
                 pred_frames: int,
                 output_units: int,
                 transform_layer: _BaseTransformLayer,
                 *args, **kwargs) -> None:

        super().__init__()

        self.d = output_units
        self.T_layer = transform_layer

        self.obs_frames = obs_frames
        self.pred_frames = pred_frames

        # Linear prediction layer
        self.linear = layers.LinearLayerND(self.obs_frames,
                                           self.pred_frames,
                                           return_full_trajectory=True)

        # Trajectory encoding (ego)
        self.te = layers.TrajEncoding(self.T_layer.Oshape[-1], self.d,
                                      torch.nn.Tanh,
                                      transform_layer=self.T_layer)

        # Linear trajectory encoding
        self.le = layers.TrajEncoding(self.T_layer.Oshape[-1], self.d,
                                      torch.nn.Tanh,
                                      transform_layer=self.T_layer)

    def forward(self, x_ego: torch.Tensor, *args, **kwargs):
        # Move the last obs point to `(0, 0)`
        ref = x_ego[..., -1:, :]
        x_ego_pure = x_ego - ref

        # Compute the linear trajectory
        traj_linear = self.linear(x_ego_pure)   # (batch, obs+pred, dim)

        # Move the linear trajectory to make it intersect with the obs trajectory
        # at the current observation moment (by moving it to (0. 0)).
        _t = self.obs_frames
        traj_linear = traj_linear - traj_linear[..., _t-1:_t, :]
        linear_fit = traj_linear[..., :_t, :]
        linear_base = traj_linear[..., _t:, :]

        # Trajectory embedding and encoding
        f_ego = self.te(x_ego_pure)

        # Linear trajectory embedding and encoding
        f_ego_linear = self.le(linear_fit)

        f_diff = f_ego - f_ego_linear    # ranged from (-2, 2)
        f_diff = f_diff / 2           # ranged from (-1 ,1)

        # Move back trajectories
        linear_fit = linear_fit + ref
        linear_base = linear_base + ref

        return f_diff, linear_fit, linear_base
