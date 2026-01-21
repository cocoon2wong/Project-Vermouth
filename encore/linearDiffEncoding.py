"""
@Author: Conghao Wong
@Date: 2026-01-05 14:35:27
@LastEditors: Conghao Wong
@LastEditTime: 2026-01-21 09:28:33
@Github: https://cocoon2wong.github.io
@Copyright 2026 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers
from qpid.utils import MAX_TYPE_NAME_LEN


class LinearDiffEncoding(torch.nn.Module):
    """
    Linear Difference Encoding Layer
    ---
    It is used to encode the difference between the observed trajectory and
    the corresponding linear (least square) trajectory of the ego agent.
    """

    def __init__(self, obs_frames: int,
                 traj_dim: int,
                 output_units: int,
                 transform_type: str,
                 encode_agent_types: bool | int = False,
                 enable_bilinear: bool = False,
                 *args, **kwargs):

        super().__init__()

        self.d = output_units
        self.obs_frames = obs_frames

        self.encode_types = encode_agent_types
        self.bilinear = enable_bilinear

        # Transform layers
        t_type, _ = layers.get_transform_layers(transform_type)
        self.tlayer = t_type((obs_frames, traj_dim))

        # Linear prediction layer
        self.linear = layers.LinearLayerND(
            obs_frames=self.obs_frames,
            pred_frames=1,      # This is actually useless
            return_full_trajectory=True
        )

        # Trajectory encoding (normal)
        self.te = layers.TrajEncoding(
            input_units=self.tlayer.Oshape[-1],
            output_units=self.d//2 if self.bilinear else self.d,
            activation=torch.nn.ReLU if self.bilinear else torch.nn.Tanh,
            transform_layer=self.tlayer,
        )

        # Trajectory encoding (linear)
        self.le = layers.TrajEncoding(
            input_units=self.tlayer.Oshape[-1],
            output_units=self.d//2 if self.bilinear else self.d,
            activation=torch.nn.ReLU if self.bilinear else torch.nn.Tanh,
            transform_layer=self.tlayer,
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
        x = x_ego[..., None, :, :, :]               # (b, 1, K, obs, dim)

        if x_nei is not None:
            x = x_ego.unsqueeze(-4)                 # (b, 1, K, obs, dim)
            x = torch.concat([x, x_nei], dim=-4)    # (b, 1+nei, K, obs, dim)
        else:
            x = x_ego

        # -------------------------
        # MARK: - Linear Prediction
        # -------------------------
        # Move the last obs point to `(0, 0)`
        x = x - x[..., -1:, :]

        # Compute linear trajectories (linear fit among observations)
        x_linear = self.linear(x)[..., :-1, :]      # (..., obs, dim)
        x_linear = x_linear - x_linear[..., -1:, :]
        x_diff = x - x_linear                       # (..., obs, dim)

        # ---------------------
        # MARK: - Dual Encoding
        # ---------------------
        fn = self.te(x)                     # (..., obs, dim)
        fl = self.le(x_linear)              # (..., obs, dim)

        # Bilinear refinement
        if self.bilinear:
            fn = self.outer(fn, fn)         # (..., obs, d/2, d/2)
            fn = self.flatten(fn)           # (..., obs, (d/2)^2)
            fn = self.outer_fc(fn)          # (..., obs, d)

            fl = self.outer(fl, fl)
            fl = self.flatten(fl)
            fl = self.outer_fc_linear(fl)   # (..., obs, d)

        # Compute difference feature
        fn = fn - fl                        # ranged from (-2, 2)
        fn = fn / 2                         # ranged from (-1 ,1)

        # --------------------
        # MARK: - Post Process
        # --------------------
        # Unpack trajectories and features
        if x_nei is not None:
            x_diff_ego = x_diff[..., 0, :, :, :]    # (batch, K, obs, dim)
            x_diff_nei = x_diff[..., 1:, :, :, :]   # (batch, nei, K, obs, dim)

            f_ego = fn[..., 0, :, :, :]             # (batch, K, obs, d)
            f_nei = fn[..., 1:, :, :, :]            # (batch, nei, K, obs, d)

        else:
            x_diff_ego = x_diff
            f_ego = fn

        # Encode types (if needed)
        if self.encode_types and (ego_types is not None):
            f_type = self.type_encoder(ego_types)[..., None, None, :]
            f_ego = f_ego + f_type

        if x_nei is not None:
            return (f_ego, x_diff_ego), (f_nei, x_diff_nei)
        else:
            return f_ego, x_diff_ego
