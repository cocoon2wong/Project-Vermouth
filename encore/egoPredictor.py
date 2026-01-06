"""
@Author: Conghao Wong
@Date: 2025-12-09 15:34:52
@LastEditors: Conghao Wong
@LastEditTime: 2026-01-05 21:27:47
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

        # Linear prediction
        self.linear_predictor = layers.LinearLayerND(
            obs_frames=self.t_h,
            pred_frames=self.t_f,
        )

        # Linear difference encoding (embedding)
        self.linear_diff = LinearDiffEncoding(
            obs_frames=self.t_h,
            traj_dim=self.d_traj,
            output_units=self.d//2,
            transform_type=transform,
            encode_agent_types=False,
            enable_bilinear=False,
        )

        # Transformer backbone
        # Shapes
        self.T_h, self.M_h = self.tlayer.Tshape
        self.T_f, self.M_f = self.itlayer.Tshape

        # Noise embedding
        self.noise_embedding = layers.TrajEncoding(
            input_units=self.d_noise,
            output_units=self.d//2,
            activation=torch.nn.Tanh
        )

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
        # Speed up inference #1: Remove all-empty neighbors
        # Compute max neighbor count within the batch
        valid_mask = get_mask(torch.sum(torch.abs(x_nei), dim=[-1, -2]))

        # Speed up inference #2: Limit neighbor numbers
        if self.capacity > 0:
            # Compute relative distance (at the last obs step)
            d = torch.norm(x_ego[..., -1:, :] - x_nei[..., -1, :],
                           p=2, dim=-1)

            # Compute the min-distance neighbor mask
            cap_mask = torch.zeros_like(d)
            cap_mask = torch.scatter(
                input=cap_mask,
                dim=-1,
                index=torch.topk(d, self.capacity, -1, largest=False).indices,
                value=1,
            )
        else:
            cap_mask = 1

        # Repeat and concat ego and neighbors' obs
        x_ego = x_ego[..., None, :, :].expand(
            *x_ego.shape[:-2],
            x_nei.shape[-3],
            *x_ego.shape[-2:],
        )

        # Compute final mask && Get neighbors to be considered
        indices = torch.nonzero(valid_mask * cap_mask, as_tuple=True)
        x_picked = torch.concat([x_ego, x_nei], dim=-2)[indices]

        # Concat ego and neighbors' trajectories into a `big batch`
        b = x_picked.shape[0]
        x_packed = torch.concat([
            x_picked[..., :self.t_h, :],    # x_ego
            x_picked[..., self.t_h:, :],    # x_nei
        ], dim=0)

        # Move the last obs point to (0, 0)
        ref = x_packed[..., -1:, :]         # (b*2, t_h, dim)
        x_packed = x_packed - ref

        # ------------------------
        # MARK: - Embed and Encode
        # ------------------------
        # Linear prediction (least squares)
        y_nei_linear = self.linear_pred(x_packed)[b:, None, :, :]

        # Encode difference features
        # Apply to both egos' and neighbors' observed trajectories
        f_diff, x_diff = self.linear_diff(x_packed)

        # ---------------------------
        # MARK: - Social Interactions
        # ---------------------------
        # NOTE: Ego predictor does not consider interactions

        # ----------------------------
        # MARK: - Transformer Backbone
        # ----------------------------

        all_predictions = []
        repeats = 1
        for _ in range(repeats):
            # Assign random ids and embedding -> (b*2, T_h, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f_diff.shape[:-1]) + [self.d_noise])
            f_z = self.noise_embedding(z.to(f_diff.device))

            # Transformer backbone -> (b*2, T_h, d)
            # Difference features as keys and queries in attention layers
            y, _ = self.T(inputs=torch.concat([f_diff, f_z], dim=-1),
                          targets=self.tlayer(x_diff),
                          training=training)

            # -------------------------------------
            # MARK: - Latency Prediction and Decode
            # -------------------------------------

            # Reverberation kernels and transform
            I = self.k1(y[:b, :, :])                # Using ego's feature
            R = self.k2(y[b:, :, :])                # Using neighbor's
            y = self.rev(y[b:, :, :], R, I)     # (b, ins, T_f, d)

            # Decode predictions
            y = self.decoder(y)         # (b, ins, T_f, M)
            y = self.itlayer(y)             # (b, ins, t_f, m)

            all_predictions.append(y)

        # Stack all outputs -> (b, ins, t_f, m)
        y_nei = torch.concat(all_predictions, dim=-3)

        # Final predictions
        y_nei = y_nei + y_nei_linear

        # Move back predictions
        y_nei = y_nei + ref[b:, None, :, :]

        # Run linear prediction for un-masked neighbors
        y = self.linear_pred(x_nei)
        y = y[..., None, :, :].expand(
            *y.shape[:-2],
            self.insights,
            *y.shape[-2:],
        )

        # Replace masked neighbors' predictions
        y = y.clone()
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
