"""
@Author: Conghao Wong
@Date: 2025-12-24 19:35:52
@LastEditors: Conghao Wong
@LastEditTime: 2026-04-08 09:27:06
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.dataset import Annotation
from qpid.model import layers, transformer

from ..layers import (KernelLayer, LinearDiffEncoding, ResonanceLayer,
                      ReverberationTransform)
from ..utils import repeat


class SocialPredictor(torch.nn.Module):
    """
    SocialPredictor
    ---
    The social predictor is a normal-sized global predictor.
    It uses a *Resonance*-like method to model social interactions, thereby
    forecasting socially aware predictions for each ego agent.
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
            output_units=self.d // 2,
            transform_type=transform,
            encode_agent_types=encode_agent_types,
        )

        # Resonance layer (for computing social interactions)
        # For observations
        self.resonance = ResonanceLayer(
            hidden_feature_dim=self.d,
            output_feature_dim=self.d // 2,
            angle_partitions=self.p,
        )

        # Concat layer for `f_ego` and `f_social`
        self.concat_fc = layers.Dense(self.d // 2 + self.d // 2,
                                      self.d // 2,
                                      torch.nn.ReLU)

        # Transformer backbone
        # Shapes
        self.T_h, self.M_h = self.tlayer.Tshape
        self.T_f, self.M_f = self.itlayer.Tshape

        # Noise embedding
        self.noise_embedding = layers.TrajEncoding(
            input_units=self.d_noise,
            output_units=self.d // 2,
            activation=torch.nn.Tanh
        )

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
        NOTE: Both `x_ego` and `x_nei` should use absolute coordinates 
        and share the same sequence length!
        """

        # ------------------------
        # MARK: - Embed and Encode
        # ------------------------
        # Linear prediction (least squares) and encode difference features
        # Apply to both the egos' and neighbors' observed trajectories
        (f_ego, d_ego), (f_nei, _) = self.linear_diff(
            x_ego=x_ego,
            x_nei=x_nei,
            ego_types=ego_types,
        )

        # ---------------------------
        # MARK: - Social Interactions
        # ---------------------------
        # Compute angle-based interactions based on their trust positions.
        # Positions used to compute interactions should be 2D mean trajectories.
        x_ego_mean = picker.get_center(x_ego.mean(dim=-3))[..., :2]
        x_nei_mean = picker.get_center(x_nei.mean(dim=-3))[..., :2]

        # See "Resonance: Learning to Predict Social-Aware Pedestrian
        # Trajectories as Co-Vibrations"
        f_social = self.resonance(
            x_ego_mean=x_ego_mean,
            x_nei_mean=x_nei_mean,
            f_ego=f_ego,
            f_nei=f_nei,
        )  # -> (batch, T_h, partitions, d/2)

        # ----------------------------
        # MARK: - Transformer Backbone
        # ----------------------------
        # Max-pool features across all insights
        f_ego = torch.max(f_ego, dim=-3)[0]

        # Pad features to keep a compatible tensor shape.
        # Original shape of `f_ego` is `(batch, T_h, d/2)`
        f_ego = repeat(f_ego, self.p, -2)

        # Original shape of `f_social` is `(batch, T_h, partitions, d/2)`
        f_social = torch.flatten(f_social, -3, -2)

        # Concatenate and fuse social features with trajectory features.
        # It serves as keys and queries in the attention layers.
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
            # Assign random IDs and embeddings -> (batch, steps, d/2)
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
            y = self.rev(y, R, G)               # (batch, K_g, T_f, d)

            # Decode predictions
            y = self.decoder(y)                 # (batch, K_g, T_f, M)
            y = self.itlayer(y)                 # (batch, K_g, t_f, m)

            all_predictions.append(y)

        # Stack all outputs -> (batch, K, t_f, m)
        return torch.concat(all_predictions, dim=-3)

    def vis_activations(self, nei_trajs: torch.Tensor,
                        original_obs_len: int,
                        ego_types: torch.Tensor | None = None,
                        vis_mode: str = '1'):
        """
        This method is only used for visualizing the feature selection after
        the max-pooling, i.e., the feature-level bias conditioning.
        Shape of the `nei_trajs` should be `(batch, nei, insights, obs, dim)`.

        `vis_mode` accepts three values:
        - `1`: Regular visualization.
        - `2`: Visualize activations of mean trajectories additionally.
        - `3`: Visualize absolute deviation instead of activations.

        Additionally, the default visualization neighbor is the zeroth neighbor
        (the ego itself) sorted by Euclidean distance. Neighbor IDs can be input
        as concatenated strings. For example, `1_3` refers to the third neighbor
        in mode 1 (Regular visualization).
        """

        from ..utils import vis_activations

        last_row_name = None

        # Process visualization mode
        vis_mode = vis_mode.replace(str(True), '1')
        settings = vis_mode.split('_')

        mode = int(settings[0])
        nei_id = int(settings[1]) if len(settings) >= 2 else 0

        # Compute distance (for sorting)
        # NOTE: `nei_trajs` are relative values to the ego
        d = torch.norm(nei_trajs[..., 0, original_obs_len - 1, :], p=2, dim=-1)

        # Select the neighbor to be computed
        idx = d.argsort(dim=-1)[:, nei_id]
        batch_idx = torch.arange(nei_trajs.shape[0]).to(d.device)
        trajs = nei_trajs[batch_idx, idx]

        if mode == 2:
            mean_traj = torch.mean(trajs, dim=-3, keepdim=True)
            trajs = torch.concat([trajs, mean_traj], dim=-3)
            last_row_name = 'Mean Rehearsal'

        # Embed and encode
        # -> (batch, insights, obs, dim)
        f_nei, _ = self.linear_diff(
            x_ego=trajs,
            ego_types=ego_types,
        )

        # Average features across the batch
        f_batch = torch.mean(f_nei, dim=0)

        # Start visualizing
        vis_activations(
            f=f_batch,
            title=f'Feature Activations (Social Bias of Neighbor #{nei_id})',
            last_row_name=last_row_name,
            deviation_mode=True if mode == 3 else False,
        )
