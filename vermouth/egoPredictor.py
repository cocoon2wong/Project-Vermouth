"""
@Author: Conghao Wong
@Date: 2025-12-09 15:34:52
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-24 10:19:11
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers, transformer
from qpid.utils import INIT_POSITION as INF
from qpid.utils import get_mask

from .reverberationTransform import KernelLayer


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
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Parameters
        self.t_h = obs_steps
        self.t_f = pred_steps

        self.d = feature_dim
        self.d_traj = traj_dim
        self.d_noise = noise_depth

        self.insights = insights

        # Simple trajectory encoder
        # Only a simple Transformer encoder, with sampled noise vector
        # It does not consider further interactions among neighbors
        self.embedding = layers.Dense(self.d_traj, self.d//2, torch.nn.Tanh)
        self.noise_embedding = layers.TrajEncoding(self.d_noise,
                                                   self.d//2,
                                                   torch.nn.Tanh)

        self.encoder = transformer.TransformerEncoder(
            num_layers=2,
            num_heads=2,
            dim_model=self.d,
            dim_forward=256,
            steps=self.t_h,
            dim_input=self.d_traj,
            dim_output=self.d_traj,
            include_top=False,
        )

        # Simple latency predictor, similar to the reverberation transform
        self.outer = layers.OuterLayer(self.t_h, self.t_h)
        self.reverberation_predictor = KernelLayer(self.d, self.d, self.t_f)
        self.insight_predictor = KernelLayer(self.d, self.d, self.insights)

        # Simple trajectory decoder
        self.decoder = layers.Dense(self.d, self.d_traj)

    def forward(self, ego_traj: torch.Tensor, nei_trajs: torch.Tensor):
        # IMPORTANT: Both `ego_traj` and `nei_trajs` should be absolute values!

        # ------------
        # Preprocesses
        # ------------
        # Speed up inference: Remove all-empty neighbors
        # Compute max neighbor count within the batch
        overall_mask = get_mask(torch.sum(torch.abs(nei_trajs), dim=[-1, -2]))
        max_valid_idx = torch.max(torch.where(overall_mask == 1)[1])
        max_valid_nei_count = max_valid_idx + 1
        cut_count = nei_trajs.shape[-3] - max_valid_nei_count

        # Cut trajectory matrix
        _nei_trajs = nei_trajs[..., :max_valid_nei_count, :, :]

        # Concat ego and neighbors' trajectories
        trajs = torch.concat([ego_traj[..., None, :, :],
                              _nei_trajs], dim=-3)

        if ((ego_traj.shape[-2] != self.t_h) or
                (_nei_trajs.shape[-2] != self.t_h)):
            raise ValueError('Wrong trajectory lengths!')

        # Move the last obs point to (0, 0)
        ref = trajs[..., -1:, :]      # (batch, nei+1, t_h, dim)
        trajs = trajs - ref

        # ----------------
        # Embed and Encode
        # ----------------
        # Encode features together
        # Including the insight feature and neighbor features
        f_embed = self.embedding(trajs)

        # Assign random ids and embedding
        z = torch.normal(mean=0, std=1,
                         size=list(f_embed.shape[:-1]) + [self.d_noise])
        z_embed = self.noise_embedding(z.to(f_embed.device))

        f_pack = self.encoder(torch.concat([f_embed, z_embed], dim=-1))

        # Unpack features
        f_insight = f_pack[..., 0, :, :]    # (batch, t_h, d)
        f_nei = f_pack[..., 1:, :, :]       # (batch, nei, t_h, d)

        # -----------------
        # Latency predictor
        # -----------------
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

        # -------------------------
        # Decoder and Postprocesses
        # -------------------------
        # Decode predictions
        pred = self.decoder(f)              # (batch, nei, insights, t_f, dim)

        # Move back predictions
        pred = pred + ref[..., 1:, None, :, :]

        # Add INF paddings for invalid neighbors
        paddings = INF * torch.ones_like(pred[..., :1, :, :, :])
        paddings = torch.repeat_interleave(paddings, cut_count, dim=-4)
        pred = torch.concat([pred, paddings], dim=-4)

        return pred
