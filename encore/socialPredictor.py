"""
@Author: Conghao Wong
@Date: 2025-12-24 19:35:52
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-30 19:50:04
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.dataset import Annotation
from qpid.model import layers, transformer

from .linearDiffEncoding import LinearDiffEncoding
from .resonanceLayer import ResonanceLayer
from .reverberationTransform import KernelLayer, ReverberationTransform


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
                 ego_pred_steps: int,
                 partitions: int,
                 generations: int,
                 traj_dim: int,
                 feature_dim: int,
                 noise_depth: int,
                 transform: str,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Parameters
        self.t_h = obs_steps
        self.t_f = pred_steps
        self.ego_t_f = ego_pred_steps

        self.d = feature_dim
        self.d_traj = traj_dim
        self.d_noise = noise_depth
        self.K_g = generations
        self.p = partitions

        # Layers
        # Transform layers
        t_type, it_type = layers.get_transform_layers(transform)
        self.tlayer = t_type((self.t_h + self.ego_t_f, self.d_traj))
        self.itlayer = it_type((self.t_f, self.d_traj))

        # Linear difference encoding (embedding)
        # For observations
        self.linear_diff = LinearDiffEncoding(
            obs_frames=self.t_h + self.ego_t_f,
            pred_frames=self.t_f,       # This is actually not used
            output_units=self.d//2,
            transform_layer=self.tlayer,
        )

        # Resonance layer (for computing social interactions)
        # For observations
        self.resonance = ResonanceLayer(
            hidden_feature_dim=self.d,
            output_feature_dim=self.d//2,
            angle_partitions=self.p,
            transform_layer=self.tlayer,
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
        x_packed = torch.concat([x_ego[..., None, :, :, :], x_nei], dim=-4)
        f_diff, x_linear, _ = self.linear_diff(x_packed)

        f_ego = f_diff[..., 0, :, :, :]    # (batch, Ks, steps, d/2)
        f_nei = f_diff[..., 1:, :, :, :]   # (batch, nei, Ks, steps, d/2)

        x_ego_diff = x_ego - x_linear[..., 0, :, :, :]
        x_ego_diff_mean = torch.mean(x_ego_diff, dim=-3)

        # ---------------------------
        # MARK: - Social Interactions
        # ---------------------------
        # Transpose shapes (batch, nei, Ks, ...) into (batch, Ks, nei, ...)
        x_nei_t = torch.transpose(x_nei, -3, -4)
        f_nei_t = torch.transpose(f_nei, -3, -4)

        # Compute social features on all `big batchs` (batch, Ks, ...)
        f_social, _ = self.resonance(
            x_ego_2d=picker.get_center(x_ego)[..., :2],
            x_nei_2d=picker.get_center(x_nei_t)[..., :2],
            f_ego=f_ego,
            f_nei=f_nei_t,
        )

        # ----------------------------
        # MARK: - Transformer Backbone
        # ----------------------------
        # "Max pool" features on all insights
        f_ego_pooled = torch.max(f_ego, dim=-3)[0]
        f_social_pooled = torch.max(f_social, dim=-4)[0]

        # Pad features to keep the compatible tensor shape
        # Original shape of `f_ego` is `(batch, T_h, d/2)`
        f_ego_pad = torch.repeat_interleave(f_ego_pooled, self.p, -2)

        # Original shape of `f_social` is `(batch, T_h, partitions, d/2)`
        f_social_pad = torch.flatten(f_social_pooled, -3, -2)

        # Concat and fuse social features with trajectory features
        # It serves as keys and queries in attention layers
        # -> `(batch, steps, d)`, where `steps = T_h * partitions`
        f = torch.concat([f_ego_pad, f_social_pad], dim=-1)
        f = self.concat_fc(f)

        # Target value for queries
        # -> (batch, T_h * partitions, M)
        X_ego_diff = self.tlayer(x_ego_diff_mean)
        X_ego_diff = torch.repeat_interleave(X_ego_diff, self.p, -2)

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
