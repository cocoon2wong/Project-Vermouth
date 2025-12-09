"""
@Author: Conghao Wong
@Date: 2025-12-09 15:34:52
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-09 19:40:35
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers

from .utils import KernelLayer


class EgoPredictor(torch.nn.Module):
    """
    EgoPredictor
    """

    def __init__(self, obs_steps: int,
                 pred_steps: int,
                 insights: int,
                 traj_dim: int,
                 feature_dim: int,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.t_h = obs_steps
        self.t_f = pred_steps

        self.d_traj = traj_dim
        self.d = feature_dim

        self.insights = insights

        # Simple trajectory predictor, similar to the reverberation transform
        self.outer = layers.OuterLayer(self.t_h, self.t_h)
        self.reverberation_predictor = KernelLayer(self.d, self.d, self.t_f)
        self.insight_predictor = KernelLayer(self.d, self.d, self.insights)

        # Simple trajectory encoder and decoder
        self.encoder = torch.nn.Sequential(
            layers.Dense(self.d_traj, self.d, torch.nn.ReLU),
            layers.Dense(self.d, self.d, torch.nn.ReLU),
            layers.Dense(self.d, self.d, torch.nn.ReLU),
            layers.Dense(self.d, self.d, torch.nn.ReLU),
            layers.Dense(self.d, self.d, torch.nn.ReLU),
            layers.Dense(self.d, self.d, torch.nn.Tanh),
        )

        self.decoder = layers.Dense(self.d, self.d_traj)

    def forward(self, ego_traj: torch.Tensor, nei_trajs: torch.Tensor):
        # Concat ego and neighbors' trajectories
        trajs = torch.concat([ego_traj[..., None, :, :],
                              nei_trajs], dim=-3)

        if ((ego_traj.shape[-2] != self.t_h) or
                (nei_trajs.shape[-2] != self.t_h)):
            raise ValueError('Wrong trajectory lengths!')

        # Move the last obs point to (0, 0)
        positions = trajs[..., -1:, :]      # (batch, nei+1, t_h, dim)
        trajs = trajs - positions

        # Encode features together
        # Including the insight feature and neighbor features
        f_pack = self.encoder(trajs)

        # Unpack features
        f_insight = f_pack[..., 0, :, :]    # (batch, t_h, d)
        f_nei = f_pack[..., 1:, :, :]       # (batch, nei, t_h, d)

        # Compute kernels
        # (batch, nei, t_h, t_f)
        rev_kernel = self.reverberation_predictor(f_nei)

        # (batch, 1, t_h, insights)
        ins_kernel = self.insight_predictor(f_insight)[..., None, :, :]

        # Predict (like reverberation transform)
        # Compute similarity
        f = f_nei                           # (batch, nei, t_h, d)
        f = torch.transpose(f, -1, -2)      # (batch, nei, d, t_h)
        f = self.outer(f, f)                # (batch, nei, d, t_h, t_h)

        # Apply the reverberation kernel
        R = rev_kernel[..., None, :, :]     # (batch, nei, 1, t_h, t_f)
        f = f @ R                           # (batch, nei, d, t_h, t_f)

        # Apply the insight kernel
        I = ins_kernel[..., None, :, :]     # (batch, 1, 1, t_h, insights)
        I = torch.transpose(I, -1, -2)      # (batch, 1, 1, insights, t_h)
        f = I @ f                           # (batch, nei, d, insights, t_f)

        # Sort dimensions
        f = torch.transpose(f, -1, -3)      # (batch, nei, t_f, insights, d)
        f = torch.transpose(f, -2, -3)      # (batch, nei, insights, t_f, d)

        # Decode predictions
        pred = self.decoder(f)              # (batch, nei, insights, t_f, dim)

        # Move back predictions
        pred = pred + positions[..., 1:, None, :, :]

        return pred
