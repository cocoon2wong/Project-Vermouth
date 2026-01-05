"""
@Author: Conghao Wong
@Date: 2026-01-05 14:35:27
@LastEditors: Conghao Wong
@LastEditTime: 2026-01-05 17:55:22
@Github: https://cocoon2wong.github.io
@Copyright 2026 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers
from qpid.model.layers.transfroms import _BaseTransformLayer
from qpid.utils import MAX_TYPE_NAME_LEN, get_mask


class GlobalEmbedding(torch.nn.Module):

    def __init__(self, obs_frames: int,
                 output_units: int,
                 transform_layer: _BaseTransformLayer,
                 encode_agent_types: bool | int = False,
                 enable_bilinear: bool = False,
                 *args, **kwargs):

        super().__init__()

        self.d = output_units
        self.T_layer = transform_layer

        self.obs_frames = obs_frames

        self.encode_types = encode_agent_types
        self.bilinear = enable_bilinear

        # Linear prediction layer
        self.linear = layers.LinearLayerND(
            obs_frames=self.obs_frames,
            pred_frames=1,      # This is actually useless
            return_full_trajectory=True
        )

        # Trajectory encoding (normal)
        self.te = layers.TrajEncoding(
            input_units=self.T_layer.Oshape[-1],
            output_units=self.d//2 if self.bilinear else self.d,
            activation=torch.nn.ReLU if self.bilinear else torch.nn.Tanh,
            transform_layer=self.T_layer
        )

        # Trajectory encoding (linear)
        self.le = layers.TrajEncoding(
            input_units=self.T_layer.Oshape[-1],
            output_units=self.d//2 if self.bilinear else self.d,
            activation=torch.nn.ReLU if self.bilinear else torch.nn.Tanh,
            transform_layer=self.T_layer
        )

        # Bilinear structure (outer product + pooling + fc)
        # See "Another vertical view: A hierarchical network for heterogeneous
        # trajectory prediction via spectrums."
        if self.bilinear:
            self.outer = layers.OuterLayer(self.d//2, self.d//2)
            self.flatten = layers.Flatten(axes_num=2)

            self.outer_fc = layers.Dense(
                input_units=(self.d//2)**2,
                output_units=self.d,
                activation=torch.nn.Tanh,
            )

            self.outer_fc_linear = layers.Dense(
                input_units=(self.d//2)**2,
                output_units=self.d,
                activation=torch.nn.Tanh
            )

        if self.encode_types:
            self.type_encoder = layers.Dense(MAX_TYPE_NAME_LEN,
                                             output_units,
                                             torch.nn.Tanh)

    def forward(self, x_ego: torch.Tensor,
                x_nei: torch.Tensor | None = None,
                ego_types: torch.Tensor | None = None,
                *args, **kwargs):

        # --------------------
        # MARK: - Preprocesses
        # --------------------
        # Concat ego and nei trajectories (if needed)
        x = x_ego[..., None, :, :, :]      # (batch, 1, K, obs, dim)

        if x_nei is not None:
            # -> (batch, N:=1+nei, K, obs, dim)
            x = torch.concat([x, x_nei], dim=-4)

        # Speed up computation: Remove all invalid trajs
        valid_mask = get_mask(torch.sum(torch.abs(x), dim=[-1, -2]))
        valid_idx = torch.nonzero(valid_mask, as_tuple=True)

        # Gather all valid trajectories into a big batch
        x_valid = x[valid_idx]
        x_valid = x_valid - x_valid[..., -1:, :]     # (bb, obs, dim)

        # -------------------------
        # MARK: - Linear Prediction
        # -------------------------
        # Compute linear trajectories (linear fit among observations)
        x_linear_valid = self.linear(x_valid)[..., :-1, :]  # (bb, obs, dim)
        x_linear_valid = x_linear_valid - x_linear_valid[..., -1:, :]
        x_diff_valid = x_valid - x_linear_valid             # (bb, obs, dim)

        # ---------------------
        # MARK: - Dual Encoding
        # ---------------------
        fn = self.te(x_valid)           # (bb, obs, dim)
        fl = self.le(x_linear_valid)    # (bb, obs, dim)

        # Bilinear refinement
        if self.bilinear:
            fn = self.outer(fn, fn)     # (bb, obs, d/2, d/2)
            fn = self.flatten(fn)       # (bb, obs, (d/2)^2)
            fn = self.outer_fc(fn)      # (bb, obs, d)

            fl = self.outer(fl, fl)
            fl = self.flatten(fl)
            fl = self.outer_fc_linear(fl)   # (bb, obs, d)

        # Compute difference feature
        fn = fn - fl                    # ranged from (-2, 2)
        fn = fn / 2                     # ranged from (-1 ,1)

        # --------------------
        # MARK: - Post Process
        # --------------------
        # Put invalid neighbors back (trajectories)
        x_diff = torch.zeros_like(x)
        x_diff[valid_idx] = x_diff_valid

        # Put invalid neighbors back (features)
        f = torch.zeros([*x.shape[:-2], *fn.shape[-2:]])
        f = f.to(x.device).to(torch.float32)
        f[valid_idx] = fn       # (batch, N:=1+nei, K, obs, d)

        # Unpack trajectories and features
        x_diff_ego = x_diff[..., 0, :, :, :]    # (batch, K, obs, dim)
        f_ego = f[..., 0, :, :, :]              # (batch, K, obs, d)

        if x_nei is not None:
            x_diff_nei = x_diff[..., 1:, :, :, :]   # (batch, nei, K, obs, dim)
            f_nei = f[..., 1:, :, :, :]             # (batch, nei, K, obs, d)

        # Encode types (if needed)
        if self.encode_types and (ego_types is not None):
            f_type = self.type_encoder(ego_types)[..., None, None, :]
            f_ego = f_ego + f_type

        if x_nei is not None:
            return (f_ego, x_diff_ego), (f_nei, x_diff_nei)
        else:
            return f_ego, x_diff_ego
