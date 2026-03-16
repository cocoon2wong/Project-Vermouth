"""
@Author: Conghao Wong
@Date: 2026-03-16 20:19:46
@LastEditors: Conghao Wong
@LastEditTime: 2026-03-16 21:03:51
@Github: https://cocoon2wong.github.io
@Copyright 2026 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.args import Args
from qpid.base import BaseManager
from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers, transformer
from qpid.training import Structure
from qpid.training.loss import l2

from .__args import EncoreArgs
from .egoLoss import EgoLoss
from .egoPredictor import EgoPredictor, LinearEgoPredictor
from .utils import repeat


class MinimumEncoreModel(Model):
    """
    Minimum Encore Model
    ---
    The *minimum* *Encore* trajectory prediction model.
    """

    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        # -----------------
        # MARK: - Init args
        # -----------------
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.args._set('output_pred_steps', 'all')
        self.enc_args = self.args.register_subargs(EncoreArgs, 'enc')
        self.e = self.enc_args  # alias

        # Static args
        self.e._set('encode_agent_types', 0)
        self.e._set('Kg', 1)
        self.e._set('use_linear', -1)
        self.e._set('use_intention_predictor', -1)
        self.e._set('use_social_predictor', -1)

        # Set model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.NEIGHBOR_TRAJ)

        # ---------------------
        # MARK: - Ego predictor
        # ---------------------
        # Length check
        if self.e.ego_t_f + self.e.ego_t_h > self.args.obs_frames:
            self.log('Wrong ego predictor settings (`ego_t_h` or `ego_t_f`)!',
                     level='error', raiseError=ValueError)

        # `ego_capacity = 0` is only used in ablations
        if self.e.ego_capacity == 0:
            ego_predictor_type = LinearEgoPredictor
        else:
            ego_predictor_type = EgoPredictor

        self.ego_predictor = ego_predictor_type(
            obs_steps=self.e.ego_t_h,
            pred_steps=self.e.ego_t_f,
            insights=self.e.insights,
            capacity=self.e.ego_capacity,
            traj_dim=self.dim,
            feature_dim=self.args.feature_dim//2,
            noise_depth=self.args.noise_depth,
            transform=self.e.T,
            compute_ego_bias=self.e.compute_ego_bias,
        )

        # -----------------------
        # MARK: - Final predictor
        # -----------------------
        # Embedding layer
        self.te = layers.TrajEncoding(
            input_units=self.dim,
            output_units=self.d//2,
            activation=torch.nn.Tanh,
        )

        # Transformer is used as a feature extractor
        self.T = transformer.Transformer(
            num_layers=4,
            num_heads=8,
            d_model=self.d,
            dff=512,
            pe_input=self.args.obs_frames + self.e.ego_t_f,
            pe_target=self.args.obs_frames + self.e.ego_t_f,
            input_vocab_size=self.dim,
            target_vocab_size=self.dim,
            include_top=False,
        )

        self.ms_fc = layers.Dense(self.d, self.args.pred_frames, torch.nn.Tanh)
        self.ms_conv = layers.GraphConv(self.d, self.d)

        # Noise encoding
        self.ie = layers.TrajEncoding(self.d_id, self.d//2, torch.nn.Tanh)

        # Decoder layers
        self.decoder_fc1 = layers.Dense(self.d, 2*self.d, torch.nn.Tanh)
        self.decoder_fc2 = layers.Dense(2*self.d, self.dim)

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        # --------------------
        # MARK: - Preprocesses
        # --------------------
        x_ego = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        x_nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Unpacked `x_nei` are relative values, move them back
        x_nei = x_nei + x_ego[..., None, -1:, :]

        # ---------------------
        # MARK: - Ego predictor
        # ---------------------
        _h = self.enc_args.ego_t_h
        _f = self.enc_args.ego_t_f

        # Training of ego predictor: on observation steps only
        if training:
            yy_nei_train = self.ego_predictor(
                x_ego=x_ego[..., -(_h + _f):-_f, :],
                x_nei=x_nei[..., -(_h + _f):-_f, :],
                training=training,
            )  # -> (batch, nei, insights, ego_t_f, dim)

        # Normal use of ego predictor
        # Also predict ego's trajectory
        x_s = self.ego_predictor(
            x_ego=x_ego[..., -_h:, :],
            x_nei=torch.concat([x_ego[..., None, -_h:, :],
                                x_nei[..., -_h:, :]], dim=-3),
            training=training,
        )  # -> (batch, nei+1, insights, ego_t_f, dim)

        # Unpack ego predictor's predictions and
        # Concat observations and short-term predictions
        x_ego_extended = torch.concat([
            repeat(x_ego[..., None, :, :], self.e.insights, -3),
            x_s[..., 0, :, :, :]
        ], dim=-2)

        # -----------------------
        # MARK: - Final predictor
        # -----------------------
        # Feature embedding and encoding -> (batch, insights, obs, d/2)
        f_ego = self.te(x_ego_extended)

        # "Max pool" features on all insights
        f_ego = torch.max(f_ego, dim=-3)[0]

        # Target value for queries
        targets = torch.mean(x_ego_extended, dim=-3)

        # It only predicts 1 trajectory for each agent
        repeats = 1
        all_predictions = []
        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f_ego.shape[:-1]) + [self.d_id])
            f_z = self.ie(z.to(f_ego.device))

            # Transformer inputs -> (batch, steps, d)
            f_final = torch.concat([f_ego, f_z], dim=-1)

            # Transformer outputs' shape is (batch, steps, d)
            f_tran, _ = self.T(inputs=f_final,
                               targets=targets,
                               training=training)

            # Generations -> (batch, pred_steps, d)
            adj = self.ms_fc(f_final)
            adj = torch.transpose(adj, -1, -2)
            f_multi = self.ms_conv(f_tran, adj)     # (batch, pred_steps, d)

            # Decode predictions
            y = self.decoder_fc1(f_multi)
            y = self.decoder_fc2(y)

            all_predictions.append(y)

        returns = [
            torch.concat(all_predictions, dim=-3)   # (batch, pred, dim)
        ]

        # Output predictions and labels to compute EgoLoss
        if training:
            returns += [
                x_nei[..., -_f:, :],
                yy_nei_train,
            ]

        return returns


class MinimumEncore(Structure):
    MODEL_TYPE = MinimumEncoreModel

    def __init__(self, args: list[str] | Args | None = None,
                 manager: BaseManager | None = None,
                 name='Train Manager'):

        super().__init__(args, manager, name)

        ver_args = self.args.register_subargs(EncoreArgs, 'enc')

        if ver_args.ego_capacity != 0:
            self.loss.set({l2: 1.0, EgoLoss: ver_args.ego_loss_rate})
