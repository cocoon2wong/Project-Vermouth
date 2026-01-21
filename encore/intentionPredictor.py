"""
@Author: Conghao Wong
@Date: 2025-12-24 19:13:28
@LastEditors: Conghao Wong
@LastEditTime: 2026-01-13 16:23:06
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers, transformer

from .linearDiffEncoding import LinearDiffEncoding
from .reverberationTransform import KernelLayer, ReverberationTransform


class IntentionPredictor(torch.nn.Module):
    """
    IntentionPredictor
    ---
    Intention predictor is a normal-sized global predictor.
    It only considers how agents' intentions changes, without considering
    further interactive behaviors among egos and neighbors.
    """

    def __init__(self, obs_steps: int,
                 pred_steps: int,
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

        # Layers
        # Transform layers
        t_type, it_type = layers.get_transform_layers(transform)
        self.tlayer = t_type((self.t_h, self.d_traj))
        self.itlayer = it_type((self.t_f, self.d_traj))

        # Linear difference encoding (embedding)
        self.linear_diff = LinearDiffEncoding(
            obs_frames=self.t_h,
            traj_dim=self.d_traj,
            output_units=self.d//2,
            transform_type=transform,
            encode_agent_types=encode_agent_types,
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
            num_layers=4,
            num_heads=8,
            d_model=self.d,
            dff=512,
            pe_input=self.T_h,
            pe_target=self.T_h,
            input_vocab_size=self.M_h,
            target_vocab_size=self.M_h,
            include_top=False,
        )

        # Reverberation kernels and reverberation transform layer
        self.k1 = KernelLayer(self.d, self.d, self.K_g)
        self.k2 = KernelLayer(self.d, self.d, self.T_f)

        self.rev = ReverberationTransform(
            historical_steps=self.T_h,
            future_steps=self.T_f,
        )

        # Final output layer
        self.decoder = layers.Dense(self.d, self.M_f)

    def forward(self, x_ego: torch.Tensor,
                repeats: int,
                ego_types: torch.Tensor | None = None,
                training=None,
                *args, **kwargs):

        # ------------------------
        # MARK: - Embed and Encode
        # ------------------------
        # Linear prediction (least squares) && Encode difference features
        f_ego, d_ego = self.linear_diff(
            x_ego=x_ego,
            ego_types=ego_types,
        )

        # ---------------------------
        # MARK: - Social Interactions
        # ---------------------------
        # NOTE: Intention predictor does not consider interactions

        # ----------------------------
        # MARK: - Transformer Backbone
        # ----------------------------
        # "Max pool" features on all insights
        f_ego = torch.max(f_ego, dim=-3)[0]

        # Target value for queries
        # -> (batch, partitions, M)
        targets = targets = self.tlayer(torch.mean(d_ego, dim=-3))

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
            y = self.rev(y, R, G)       # (batch, K_g, T_f, d)

            # Decode predictions
            y = self.decoder(y)         # (batch, K_g, T_f, M)
            y = self.itlayer(y)         # (batch, K_g, t_f, m)

            all_predictions.append(y)

        # Stack all outputs -> (batch, K, t_f, m)
        return torch.concat(all_predictions, dim=-3)
