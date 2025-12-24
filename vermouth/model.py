"""
@Author: Conghao Wong
@Date: 2025-12-02 11:10:53
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-24 10:03:57
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.args import Args
from qpid.base import BaseManager
from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers, transformer
from qpid.training import Structure
from qpid.training.loss import l2

from .__args import VermouthArgs
from .egoLoss import EgoLoss
from .egoPredictor import EgoPredictor
from .linearDiffEncoding import LinearDiffEncoding
from .resonanceLayer import ResonanceLayer
from .reverberationTransform import KernelLayer, ReverberationTransform


class VermouthModel(Model):
    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.args._set('output_pred_steps', 'all')
        self.ver_args = self.args.register_subargs(VermouthArgs, 'ver')
        self.v = self.ver_args  # alias

        # Set model inputs and labels
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.NEIGHBOR_TRAJ)

        # Layers
        # Transform layers
        t_type, it_type = layers.get_transform_layers(self.ver_args.T)
        self.tlayer = t_type((self.args.obs_frames, self.dim))
        self.itlayer = it_type((self.args.pred_frames, self.dim))

        # Ego predictor
        if self.v.ego_t_f + self.v.ego_t_h > self.args.obs_frames:
            self.log('Wrong ego predictor settings (`ego_t_h` or `ego_t_f`)!',
                     level='error', raiseError=ValueError)

        self.ego_predictor = EgoPredictor(
            obs_steps=self.v.ego_t_h,
            pred_steps=self.v.ego_t_f,
            insights=self.v.insights,
            traj_dim=self.dim,
            feature_dim=self.args.feature_dim//2,
            noise_depth=self.args.noise_depth,
        )

        # Linear difference encoding
        self.linear_diff = LinearDiffEncoding(
            obs_frames=self.args.obs_frames,
            pred_frames=self.args.pred_frames,
            output_units=self.d//2,
            transform_layer=self.tlayer,
        )

        # Resonance layer (for computing social interactions)
        self.resonance = ResonanceLayer(
            hidden_feature_dim=self.d,
            output_feature_dim=self.d//2,
            angle_partitions=self.ver_args.partitions,
            transform_layer=self.tlayer,
        )

        # Concat layer for `f_ego` and `f_social`
        self.concat_fc = layers.Dense(self.d//2 + self.d//2,
                                      self.d//2,
                                      torch.nn.ReLU)

        # Transformer backbone
        # Shapes
        self.K_g = self.ver_args.Kg
        self.T_h, self.M_h = self.tlayer.Tshape
        self.T_f, self.M_f = self.itlayer.Tshape

        # Noise encoding
        self.d_noise = self.args.noise_depth
        self.ie = layers.TrajEncoding(self.d_noise, self.d//2, torch.nn.Tanh)

        # Transformer as the feature extractor
        self.p = self.ver_args.partitions
        self.T = transformer.Transformer(
            num_layers=5,
            d_model=self.d,
            num_heads=8,
            dff=512,
            input_vocab_size=self.M_h,
            target_vocab_size=self.M_h,
            pe_input=self.T_h * self.p,
            pe_target=self.T_h * self.p,
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

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        # -------------
        # Unpack inputs
        # -------------
        x_ego = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        x_nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Unpacked `x_nei` are relative values, move them back
        x_nei = x_nei + x_ego[..., None, -1:, :]

        # -------------
        # Ego predictor
        # -------------
        _h = self.ver_args.ego_t_h
        _f = self.ver_args.ego_t_f

        # Training of ego predictor: on observation steps only
        if training:
            yy_nei_train = self.ego_predictor(
                ego_traj=x_ego[..., -(_h + _f):-_f, :],
                nei_trajs=x_nei[..., -(_h + _f):-_f, :],
            )  # -> (batch, nei, insights, ego_t_f, dim)

        # Normal use of ego predictor
        yy_nei = self.ego_predictor(
            ego_traj=x_ego[..., -_h:, :],
            nei_trajs=x_nei[..., -_h:, :],
        )  # -> (batch, nei, insights, ego_t_f, dim)

        # -> (batch, nei, ego_t_f, dim)
        yy_nei = torch.mean(yy_nei, dim=-3)

        # "Mess Up" the time axis
        x_nei_old = x_nei
        x_nei = torch.concat([x_nei[..., -_h:, :], yy_nei], dim=-2)

        # -----------------------------------------
        # Linear-Difference Encoding and Prediction
        # -----------------------------------------
        # Linear prediction (least squares) && Encode difference features
        # Apply to all ego and neighbors' observed trajectories
        x_packed = torch.concat([x_ego[..., None, :, :], x_nei], dim=-3)
        f_diff, x_linear, y_linear = self.linear_diff(x_packed)

        f_ego = f_diff[..., 0, :, :]    # (batch, steps, d/2)
        f_nei = f_diff[..., 1:, :, :]   # (batch, nei, steps, d/2)

        x_ego_diff = x_ego - x_linear[..., 0, :, :]
        y_ego_linear = y_linear[..., None, 0, :, :]

        # -------------------
        # Social Interactions
        # -------------------
        f_social, _ = self.resonance(
            x_ego_2d=self.picker.get_center(x_ego)[..., :2],
            x_nei_2d=self.picker.get_center(x_nei)[..., :2],
            f_ego=f_ego,
            f_nei=f_nei,
        )

        # --------------------
        # Transformer Backbone
        # --------------------
        # Pad features to keep the compatible tensor shape
        # Original shape of `f_ego` is `(batch, T_h, d/2)`
        f_ego_pad = torch.repeat_interleave(f_ego, self.p, -2)

        # Original shape of `f_social` is `(batch, T_h, partitions, d/2)`
        f_social_pad = torch.flatten(f_social, -3, -2)

        # Concat and fuse social features with trajectory features
        # It serves as keys and queries in attention layers
        # -> `(batch, steps, d)`, where `steps = T_h * partitions`
        f = torch.concat([f_ego_pad, f_social_pad], dim=-1)
        f = self.concat_fc(f)

        # Target value for queries
        # -> (batch, T_h * partitions, M)
        traj_targets = self.tlayer(x_ego_diff)
        traj_targets = torch.repeat_interleave(traj_targets, self.p, -2)

        all_predictions = []
        repeats = self.args.K_train if training else self.args.K
        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f.shape[:-1]) + [self.d_noise])
            re_f_z = self.ie(z.to(f.device))

            # -> (batch, steps, d)
            re_f_final = torch.concat([f, re_f_z], dim=-1)

            # Transformer backbone -> (batch, steps, d)
            f_tran, _ = self.T(inputs=re_f_final,
                               targets=traj_targets,
                               training=training)

            # -----------------------------
            # Latency Prediction and Decode
            # -----------------------------
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

        # Final predictions
        y_ego = y_ego + y_ego_linear

        returns = [
            y_ego,
            # yy_nei,
        ]

        if training:
            returns += [
                x_nei_old[..., -_f:, :],
                yy_nei_train,
            ]

        return returns


class Vermouth(Structure):
    MODEL_TYPE = VermouthModel

    def __init__(self, args: list[str] | Args | None = None,
                 manager: BaseManager | None = None,
                 name='Train Manager'):

        super().__init__(args, manager, name)

        self.ver_args = self.args.register_subargs(VermouthArgs, 'ver')

        self.loss.set({l2: 1.0, EgoLoss: self.ver_args.ego_loss_rate})
