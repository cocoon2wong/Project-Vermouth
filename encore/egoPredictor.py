"""
@Author: Conghao Wong
@Date: 2025-12-09 15:34:52
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-24 15:49:43
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers, transformer
from qpid.utils import INIT_POSITION as INF
from qpid.utils import get_mask

from .linearDiffEncoding import LinearDiffEncoding
from .reverberationTransform import KernelLayer, ReverberationTransform


class EgoPredictor(torch.nn.Module):
    """
    EgoPredictor
    ---
    Ego predictor is a small predictor hosted by each ego agent.
    It shares similar but simplified structures as the outer predictor,
    with a simple Transformer backbone (self-attention only) and no further
    interaction-modeling components.
    """

    def __init__(self, obs_steps: int,
                 pred_steps: int,
                 insights: int,
                 traj_dim: int,
                 feature_dim: int,
                 noise_depth: int,
                 transform: str,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Parameters
        self.t_h = obs_steps
        self.t_f = pred_steps

        self.d = feature_dim
        self.d_traj = traj_dim
        self.d_noise = noise_depth
        self.insights = insights

        # Layers
        # Transform layers
        t_type, it_type = layers.get_transform_layers(transform)
        self.tlayer = t_type((self.t_h, self.d_traj))
        self.itlayer = it_type((self.t_f, self.d_traj))

        # Simple trajectory encoder
        # Only a simple Transformer encoder, with sampled noise vector
        # It does not consider further interactions among neighbors

        # Linear difference encoding (embedding)
        self.linear_diff = LinearDiffEncoding(
            obs_frames=self.t_h,
            pred_frames=self.t_f,
            output_units=self.d//2,
            transform_layer=self.tlayer,
        )

        # Transformer backbone
        # Shapes
        self.T_h, self.M_h = self.tlayer.Tshape
        self.T_f, self.M_f = self.itlayer.Tshape

        # Noise embedding
        self.noise_embedding = layers.TrajEncoding(self.d_noise,
                                                   self.d//2,
                                                   torch.nn.Tanh)

        # Transformer as the feature extractor
        self.T = transformer.Transformer(
            num_layers=2,
            num_heads=2,
            d_model=self.d,
            dff=256,
            pe_input=self.T_h,
            pe_target=self.T_h,
            input_vocab_size=self.M_h,
            target_vocab_size=self.M_h,
            include_top=False,
        )

        # Reverberation kernels and reverberation transform layer
        self.k1 = KernelLayer(self.d, self.d, self.insights)
        self.k2 = KernelLayer(self.d, self.d, self.T_f)

        self.rev = ReverberationTransform(
            historical_steps=self.T_h,
            future_steps=self.T_f,
        )

        # Final output layer
        self.decoder = layers.Dense(self.d, self.M_f)

    def forward(self, ego_traj: torch.Tensor,
                nei_trajs: torch.Tensor,
                training=None,
                mask=None,
                *args, **kwargs):
        """
        Run the ego predictor.
        IMPORTANT: Both `ego_traj` and `nei_trajs` should be absolute values, 
        and share the same sequence length!
        """

        # --------------------
        # MARK: - Preprocesses
        # --------------------
        # Speed up inference: Remove all-empty neighbors
        # Compute max neighbor count within the batch
        overall_mask = get_mask(torch.sum(torch.abs(nei_trajs), dim=[-1, -2]))
        max_valid_idx = torch.max(torch.where(overall_mask == 1)[1])
        max_valid_nei_count = max_valid_idx + 1
        cut_count = nei_trajs.shape[-3] - max_valid_nei_count

        # Cut trajectory matrix
        _nei_trajs = nei_trajs[..., :max_valid_nei_count, :, :]

        # Concat ego and neighbors' trajectories
        x_packed = torch.concat([ego_traj[..., None, :, :],
                                 _nei_trajs], dim=-3)

        if ((ego_traj.shape[-2] != self.t_h) or
                (_nei_trajs.shape[-2] != self.t_h)):
            raise ValueError('Wrong trajectory lengths!')

        # Move the last obs point to (0, 0)
        ref = x_packed[..., -1:, :]         # (batch, nei+1, T_h, dim)
        x_packed = x_packed - ref

        # ------------------------
        # MARK: - Embed and Encode
        # ------------------------
        # Linear prediction (least squares) && Encode difference features
        # Apply to both egos' and neighbors' observed trajectories
        f_diff, x_linear, y_linear = self.linear_diff(x_packed)

        x_diff = x_packed - x_linear
        y_nei_linear = y_linear[..., 1:, None, :, :]

        # ---------------------------
        # MARK: - Social Interactions
        # ---------------------------
        # NOTE: Ego predictor does not consider interactions

        # ----------------------------
        # MARK: - Transformer Backbone
        # ----------------------------
        # Difference features as keys and queries in attention layers
        f = f_diff

        # Target value for queries
        # -> (batch, nei, T_h, M)
        X_diff = self.tlayer(x_diff)

        all_predictions = []
        repeats = 1
        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, nei, T_h, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f.shape[:-1]) + [self.d_noise])
            f_z = self.noise_embedding(z.to(f.device))

            # -> (batch, nei, T_h, d)
            f_final = torch.concat([f, f_z], dim=-1)

            # Transformer backbone -> (batch, nei, T_h, d)
            f_tran, _ = self.T(inputs=f_final,
                               targets=X_diff,
                               training=training)

            # -------------------------------------
            # MARK: - Latency Prediction and Decode
            # -------------------------------------
            # Unpack features
            f_ego = f_tran[..., :1, :, :]
            f_nei = f_tran[..., 1:, :, :]

            # Reverberation kernels and transform
            I = self.k1(f_ego)
            R = self.k2(f_nei)
            f_rev = self.rev(f_nei, R, I)   # (batch, nei, ins, T_f, d)

            # Decode predictions
            y = self.decoder(f_rev)         # (batch, nei, ins, T_f, M)
            y = self.itlayer(y)             # (batch, nei, ins, t_f, m)

            all_predictions.append(y)

        # Stack all outputs -> (batch, nei, ins, t_f, m)
        y_nei = torch.concat(all_predictions, dim=-3)

        # Final predictions
        y_nei = y_nei + y_nei_linear

        # Move back predictions
        y_nei = y_nei + ref[..., 1:, None, :, :]

        # Add INF paddings for invalid neighbors
        paddings = INF * torch.ones_like(y_nei[..., :1, :, :, :])
        paddings = torch.repeat_interleave(paddings, cut_count, dim=-4)
        y_nei = torch.concat([y_nei, paddings], dim=-4)

        return y_nei
