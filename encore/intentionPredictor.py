"""
@Author: Conghao Wong
@Date: 2025-12-24 19:13:28
@LastEditors: Conghao Wong
@LastEditTime: 2026-01-04 15:48:53
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
                 ego_pred_steps: int,
                 generations: int,
                 traj_dim: int,
                 feature_dim: int,
                 noise_depth: int,
                 transform: str,
                 *args, **kwargs):

        super().__init__()

        # Parameters
        self.t_h = obs_steps
        self.t_f = pred_steps
        self.ego_t_f = ego_pred_steps

        self.d = feature_dim
        self.d_traj = traj_dim
        self.d_noise = noise_depth
        self.K_g = generations

        # Layers
        # Transform layers
        t_type, it_type = layers.get_transform_layers(transform)
        self.tlayer = t_type((self.t_h + self.ego_t_f, self.d_traj))
        self.itlayer = it_type((self.t_f, self.d_traj))

        # Linear difference encoding (embedding)
        self.linear_diff = LinearDiffEncoding(
            obs_frames=self.t_h + self.ego_t_f,
            pred_frames=self.t_f,       # This is actually not used
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
                training=None,
                mask=None,
                *args, **kwargs):

        # ------------------------
        # MARK: - Embed and Encode
        # ------------------------
        # Linear prediction (least squares) && Encode difference features
        f_ego_diff, x_ego_linear, _ = self.linear_diff(x_ego)

        x_ego_diff = x_ego - x_ego_linear
        x_ego_diff_mean = torch.mean(x_ego_diff, dim=-3)

        # ---------------------------
        # MARK: - Social Interactions
        # ---------------------------
        # NOTE: Intention predictor does not consider interactions

        # ----------------------------
        # MARK: - Transformer Backbone
        # ----------------------------
        # "Max pool" features on all insights
        f_ego_pooled = torch.max(f_ego_diff, dim=-3)[0]

        # Difference features as keys and queries in attention layers
        f = f_ego_pooled

        # Target value for queries
        # -> (batch, T_h, M)
        X_ego_diff = self.tlayer(x_ego_diff_mean)

        all_predictions = []
        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f.shape[:-1]) + [self.d_noise])
            f_z = self.noise_embedding(z.to(f.device))

            # -> (batch, steps, d)
            f_final = torch.concat([f, f_z], dim=-1)

            # Transformer backbone -> (batch, steps, d)
            f_tran, _ = self.T(inputs=f_final,
                               targets=X_ego_diff,
                               training=training)

            # -------------------------------------
            # MARK: - Latency Prediction and Decode
            # -------------------------------------
            # Reverberation kernels and transform
            G = self.k1(f_tran)
            R = self.k2(f_tran)
            f_rev = self.rev(f_tran, R, G)          # (batch, K_g, T_f, d)

            # Decode predictions
            y = self.decoder(f_rev)                 # (batch, K_g, T_f, M)
            y = self.itlayer(y)                     # (batch, K_g, t_f, m)

            all_predictions.append(y)

        # Stack all outputs -> (batch, K, t_f, m)
        y_ego = torch.concat(all_predictions, dim=-3)

        return y_ego
