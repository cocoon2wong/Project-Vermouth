"""
@Author: Conghao Wong
@Date: 2025-12-24 19:35:52
@LastEditors: Conghao Wong
@LastEditTime: 2026-01-13 16:23:54
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.dataset import Annotation
from qpid.model import layers, transformer

from .linearDiffEncoding import LinearDiffEncoding
from .resonanceLayer import ResonanceLayer
from .reverberationTransform import KernelLayer, ReverberationTransform
from .utils import repeat


class SocialPredictor(torch.nn.Module):
    """
    SocialPredictor
    ---
    Social predictor is a normal-sized global predictor.
    It use a *Resonance*-like way to model social interactions, therefore
    forecasting social-aware predictions for each ego.
    """

    def __init__(self, obs_steps: int,
                 pred_steps: int,
                 partitions: int,
                 generations: int,
                 traj_dim: int,
                 feature_dim: int,
                 noise_depth: int,
                 transform: str,
                 encode_agent_types: int | bool = False,
                 *args, **kwargs):

        super().__init__()

        # Parameters
        self.t_h = obs_steps
        self.t_f = pred_steps

        self.d = feature_dim
        self.d_traj = traj_dim
        self.d_noise = noise_depth
        self.K_g = generations
        self.p = partitions

        # Layers
        # Transform layers
        t_type, it_type = layers.get_transform_layers(transform)
        self.tlayer = t_type((self.t_h, self.d_traj))
        self.itlayer = it_type((self.t_f, self.d_traj))

        # Linear difference encoding (embedding)
        # For observations
        self.linear_diff = LinearDiffEncoding(
            obs_frames=self.t_h,
            traj_dim=self.d_traj,
            output_units=self.d//2,
            transform_type=transform,
            encode_agent_types=encode_agent_types,
        )

        # Resonance layer (for computing social interactions)
        # For observations
        self.resonance = ResonanceLayer(
            hidden_feature_dim=self.d,
            output_feature_dim=self.d//2,
            angle_partitions=self.p,
        )

        # Concat layer for `f_ego` and `f_social`
        self.concat_fc = layers.Dense(self.d//2 + self.d//2,
                                      self.d//2,
                                      torch.nn.ReLU)

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
            num_heads=8,
            d_model=self.d,
            dff=512,
            pe_input=self.T_h * self.p,
            pe_target=self.T_h * self.p,
            input_vocab_size=self.M_h,
            target_vocab_size=self.M_h,
            include_top=False,
        )

        # Reverberation kernels and reverberation transform layer
        self.k1 = KernelLayer(self.d, self.d, self.K_g)
        self.k2 = KernelLayer(self.d, self.d, self.T_f)

        self.rev = ReverberationTransform(
            historical_steps=self.T_h * self.p,
            future_steps=self.T_f,
        )

        # Final output layer
        self.decoder = layers.Dense(self.d, self.M_f)

    def forward(self, x_ego: torch.Tensor,
                x_nei: torch.Tensor,
                repeats: int,
                picker: Annotation,
                ego_types: torch.Tensor | None = None,
                training=None,
                mask=None,
                *args, **kwargs):
        """
        NOTE: Both `ego_traj` and `nei_trajs` should be absolute values, 
        and share the same sequence length!
        """

        # ------------------------
        # MARK: - Embed and Encode
        # ------------------------
        # Linear prediction (least squares) && Encode difference features
        # Apply to both egos' and neighbors' observed trajectories
        (f_ego, d_ego), (f_nei, _) = self.linear_diff(
            x_ego=x_ego,
            x_nei=x_nei,
            ego_types=ego_types,
        )

        # ---------------------------
        # MARK: - Social Interactions
        # ---------------------------
        # Transpose shapes (batch, nei, Ks, ...) into (batch, Ks, nei, ...)
        x_nei = torch.transpose(x_nei, -3, -4)
        f_nei = torch.transpose(f_nei, -3, -4)

        # Compute social features on all `big batchs` (batch, Ks, ...)
        f_social = self.resonance(
            x_ego_2d=picker.get_center(x_ego)[..., :2],
            x_nei_2d=picker.get_center(x_nei)[..., :2],
            f_ego=f_ego,
            f_nei=f_nei,
        )

        # ----------------------------
        # MARK: - Transformer Backbone
        # ----------------------------
        # "Max pool" features on all insights
        f_ego = torch.max(f_ego, dim=-3)[0]
        f_social = torch.max(f_social, dim=-4)[0]

        # Pad features to keep the compatible tensor shape
        # Original shape of `f_ego` is `(batch, T_h, d/2)`
        f_ego = repeat(f_ego, self.p, -2)

        # Original shape of `f_social` is `(batch, T_h, partitions, d/2)`
        f_social = torch.flatten(f_social, -3, -2)

        # Concat and fuse social features with trajectory features
        # It serves as keys and queries in attention layers
        # -> `(batch, steps, d)`, where `steps = T_h * partitions`
        f_ego = torch.concat([f_ego, f_social], dim=-1)
        f_ego = self.concat_fc(f_ego)

        # Target value for queries
        # -> (batch, T_h * partitions, M)
        targets = self.tlayer(torch.mean(d_ego, dim=-3))
        targets = repeat(targets, self.p, -2)

        # Make random predictions
        all_predictions = []
        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f_ego.shape[:-1]) + [self.d_noise])
            f_z = self.noise_embedding(z.to(f_ego.device))

            # Transformer backbone -> (batch, steps, d)
            y, _ = self.T(inputs=torch.concat([f_ego, f_z], dim=-1),
                          targets=targets,
                          training=training)

            # -------------------------------------
            # MARK: - Latency Prediction and Decode
            # -------------------------------------
            # Reverberation kernels and transform
            G = self.k1(y)
            R = self.k2(y)
            y = self.rev(y, R, G)          # (batch, K_g, T_f, d)

            # Decode predictions
            y = self.decoder(y)                 # (batch, K_g, T_f, M)
            y = self.itlayer(y)                     # (batch, K_g, t_f, m)

            all_predictions.append(y)

        # Stack all outputs -> (batch, K, t_f, m)
        return torch.concat(all_predictions, dim=-3)
