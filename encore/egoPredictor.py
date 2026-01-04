"""
@Author: Conghao Wong
@Date: 2025-12-09 15:34:52
@LastEditors: Conghao Wong
@LastEditTime: 2026-01-04 15:51:29
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers, transformer
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
                 capacity: int,
                 traj_dim: int,
                 feature_dim: int,
                 noise_depth: int,
                 transform: str,
                 *args, **kwargs):

        super().__init__()

        # Parameters
        self.t_h = obs_steps
        self.t_f = pred_steps

        self.d = feature_dim
        self.d_traj = traj_dim
        self.d_noise = noise_depth
        self.insights = insights
        self.capacity = capacity

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
            dff=128,
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

        # Linear prediction layer (for out-of-capacity neighbors)
        self.linear_pred = layers.LinearLayerND(
            obs_frames=self.t_h,
            pred_frames=self.t_f,
        )

    def forward(self, x_ego: torch.Tensor,
                x_nei: torch.Tensor,
                training=None,
                mask=None,
                *args, **kwargs):
        """
        Run the ego predictor.
        NOTE: Both `ego_traj` and `nei_trajs` should be absolute values, 
        and share the same sequence length!
        """

        if ((x_ego.shape[-2] != self.t_h) or
                (x_nei.shape[-2] != self.t_h)):
            raise ValueError('Wrong trajectory lengths!')

        # --------------------
        # MARK: - Preprocesses
        # --------------------
        # Repeat and concat ego and neighbors' obs
        max_nei_count = x_nei.shape[-3]
        _x_ego = x_ego[..., None, :, :].expand(
            *x_ego.shape[:-2],
            max_nei_count,
            *x_ego.shape[-2:],
        )

        _x = torch.concat([_x_ego, x_nei], dim=-2)

        # Speed up inference #1: Remove all-empty neighbors
        # Compute max neighbor count within the batch
        valid_mask = get_mask(torch.sum(torch.abs(x_nei), dim=[-1, -2]))

        # Speed up inference #2: Limit neighbor numbers
        if self.capacity > 0:
            # Compute relative distance (at the last obs step)
            d = torch.norm(x_ego[..., -1:, :] - x_nei[..., -1, :],
                           p=2, dim=-1)
            idx = torch.topk(d, self.capacity, dim=-1, largest=False).indices

            # Compute the min-distance neighbor mask
            cap_mask = torch.zeros_like(d)
            cap_mask = torch.scatter(cap_mask, -1, idx, 1)
        else:
            cap_mask = 1

        # Compute final mask
        final_mask = valid_mask * cap_mask

        # Get neighbors to be considered
        indices = torch.nonzero(final_mask, as_tuple=True)
        x_picked = _x[indices]

        _x_ego = x_picked[..., :self.t_h, :]
        _x_nei = x_picked[..., self.t_h:, :]

        # Concat ego and neighbors' trajectories into a `big batch`
        b = _x_ego.shape[0]
        x_packed = torch.concat([_x_ego, _x_nei], dim=0)

        # Move the last obs point to (0, 0)
        ref = x_packed[..., -1:, :]         # (b*2, t_h, dim)
        x_packed = x_packed - ref

        # ------------------------
        # MARK: - Embed and Encode
        # ------------------------
        # Linear prediction (least squares) && Encode difference features
        # Apply to both egos' and neighbors' observed trajectories
        f_diff, x_linear, y_linear = self.linear_diff(x_packed)

        x_diff = x_packed - x_linear
        y_nei_linear = y_linear[b:, None, :, :]

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
        # -> (b*2, T_h, M)
        X_diff = self.tlayer(x_diff)

        all_predictions = []
        repeats = 1
        for _ in range(repeats):
            # Assign random ids and embedding -> (b*2, T_h, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f.shape[:-1]) + [self.d_noise])
            f_z = self.noise_embedding(z.to(f.device))

            # -> (b*2, T_h, d)
            f_final = torch.concat([f, f_z], dim=-1)

            # Transformer backbone -> (b*2, T_h, d)
            f_tran, _ = self.T(inputs=f_final,
                               targets=X_diff,
                               training=training)

            # -------------------------------------
            # MARK: - Latency Prediction and Decode
            # -------------------------------------
            # Unpack features
            f_ego = f_tran[:b, :, :]
            f_nei = f_tran[b:, :, :]

            # Reverberation kernels and transform
            I = self.k1(f_ego)
            R = self.k2(f_nei)
            f_rev = self.rev(f_nei, R, I)   # (b, ins, T_f, d)

            # Decode predictions
            y = self.decoder(f_rev)         # (b, ins, T_f, M)
            y = self.itlayer(y)             # (b, ins, t_f, m)

            all_predictions.append(y)

        # Stack all outputs -> (b, ins, t_f, m)
        y_nei = torch.concat(all_predictions, dim=-3)

        # Final predictions
        y_nei = y_nei + y_nei_linear

        # Move back predictions
        y_nei = y_nei + ref[b:, None, :, :]

        # Run linear prediction for un-masked neighbors
        y_nei_base = self.linear_pred(x_nei)
        y_nei_base = y_nei_base[..., None, :, :].expand(
            *y_nei_base.shape[:-2],
            self.insights,
            *y_nei_base.shape[-2:],
        )

        # Replace masked neighbors' predictions
        y = y_nei_base.clone()
        y[indices] = y_nei

        return y


class LinearEgoPredictor(torch.nn.Module):
    """
    LinearEgoPredictor
    ---
    This is the simple linear-prediction-based ego predictor,
    only used for ablations and discussions.
    """

    def __init__(self, obs_steps: int,
                 pred_steps: int,
                 insights: int,
                 *args, **kwargs):

        super().__init__()

        # Parameters
        self.t_h = obs_steps
        self.t_f = pred_steps
        self.insights = insights

        # Layers
        self.linear_pred = layers.LinearLayerND(
            obs_frames=self.t_h,
            pred_frames=self.t_f,
        )

    def forward(self, x_nei: torch.Tensor, *args, **kwargs):

        y_nei = self.linear_pred(x_nei)
        y_nei = y_nei[..., None, :, :].expand(
            *y_nei.shape[:-2],
            self.insights,
            *y_nei.shape[-2:],
        )

        return y_nei
