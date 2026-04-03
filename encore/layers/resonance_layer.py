"""
@Author: Conghao Wong
@Date: 2025-12-23 10:22:15
@LastEditors: Conghao Wong
@LastEditTime: 2026-01-21 09:24:09
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import torch

from qpid.model import layers
from qpid.utils import get_mask


class ResonanceLayer(torch.nn.Module):
    """
    Resonance Layer
    ---
    Compute the spectral similarities of *pure* observed trajectories between
    the ego agent and all other neighboring agents. An angle-based pooling will
    be applied to gather these spectral similarities within limited angle-based
    partitions.
    """

    def __init__(self, hidden_feature_dim: int,
                 output_feature_dim: int,
                 angle_partitions: int,
                 *args, **kwargs) -> None:

        super().__init__()

        # Variables and Settings
        self.p = angle_partitions
        self.d = output_feature_dim
        self.d_h = hidden_feature_dim

        # Layers
        # Position encoding (not positional encoding)
        self.ce = layers.TrajEncoding(2, self.d//2, torch.nn.ReLU)

        # Encoding layers (for resonance features)
        self.fc1 = layers.Dense(self.d, self.d_h, torch.nn.ReLU)
        self.fc2 = layers.Dense(self.d_h, self.d_h, torch.nn.ReLU)
        self.fc3 = layers.Dense(self.d_h, self.d//2, torch.nn.ReLU)

    def forward(self, x_ego_mean: torch.Tensor,
                x_nei_mean: torch.Tensor,
                f_ego: torch.Tensor,
                f_nei: torch.Tensor):

        # Compute meta resonance features (for each neighbor)
        # -> (batch, N, insights, obs, d)
        f_nei = f_ego.unsqueeze(-4) * f_nei

        # "Max pool" features on all insights
        f_nei = torch.max(f_nei, dim=-3)[0]

        # Shape of the final output `f_re`: (batch, N, obs, d/2)
        f_nei = self.fc3(self.fc2(self.fc1(f_nei)))

        # Compute positional information in a SocialCircle-like way
        # Time-resolution of the used transform
        t_r = int(np.ceil(x_ego_mean.shape[-2] / f_ego.shape[-2]))

        p_nei = x_nei_mean - x_ego_mean[..., None, :, :]
        p_nei = p_nei[..., ::t_r, :]

        # Compute distances and angles (for all neighbors)
        f_distance = torch.norm(p_nei, dim=-1)
        f_angle = torch.atan2(p_nei[..., 0], p_nei[..., 1])
        f_angle = f_angle % (2 * np.pi)

        # Partitioning
        partition_indices = f_angle / (2*np.pi/self.p)
        partition_indices = partition_indices.to(torch.int32)

        # Mask neighbors
        # Compute resonance features on all steps and partitions
        nei_mask = get_mask(torch.sum(torch.abs(p_nei),
                                      dim=-1), torch.int32)

        # Remove egos from neighbors (the self-neighbors)
        non_self_mask = (f_distance > 0.005).to(torch.int32)
        final_mask = nei_mask * non_self_mask

        # Angle-based pooling
        pos_list: list[list[torch.Tensor]] = []
        y = []
        partition_indices = (partition_indices * final_mask +
                             -1 * (1 - final_mask))

        for _p in range(self.p):
            _mask = (partition_indices == _p).to(torch.float32)
            _mask_count = torch.sum(_mask, dim=-2)

            n = _mask_count + 0.0001

            pos_list.append([])
            pos_list[-1].append(torch.sum(f_distance * _mask, dim=-2) / n)
            pos_list[-1].append(torch.sum(f_angle * _mask, dim=-2) / n)
            y.append(torch.sum(f_nei * _mask[..., None], dim=-3) /
                     n[..., None])

        # Stack all partitions
        # (batch, steps, partitions, 2)
        positions = torch.stack([torch.stack(i, dim=-1) for i in pos_list],
                                dim=-2)

        # Concat resonance features -> (batch, steps, partition, d)
        y = torch.concat([
            torch.stack(y, dim=-2),     # (batch, steps, partitions, d/2)
            self.ce(positions)          # (batch, steps, partition, d/2)
        ], dim=-1)

        return y
