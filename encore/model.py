"""
@Author: Conghao Wong
@Date: 2025-12-02 11:10:53
@LastEditors: Conghao Wong
@LastEditTime: 2026-01-06 11:20:59
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.args import Args
from qpid.base import BaseManager
from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers
from qpid.training import Structure
from qpid.training.loss import l2

from .__args import EncoreArgs
from .egoLoss import EgoLoss
from .egoPredictor import EgoPredictor, LinearEgoPredictor
from .intentionPredictor import IntentionPredictor
from .socialPredictor import SocialPredictor
from .utils import repeat


class EncoreModel(Model):
    """
    EncoreModel
    ---
    The *Encore* trajectory prediction model.
    """

    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.args._set('output_pred_steps', 'all')
        self.enc_args = self.args.register_subargs(EncoreArgs, 'enc')
        self.e = self.enc_args  # alias

        # Set model inputs
        # Types of agents are only used in complex scenes
        # For other datasets, keep it disabled (through the arg)
        if not self.e.encode_agent_types:
            self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                            INPUT_TYPES.NEIGHBOR_TRAJ)
        else:
            self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                            INPUT_TYPES.NEIGHBOR_TRAJ,
                            INPUT_TYPES.AGENT_TYPES)

        # Predictors
        # Linear predictor
        self.linear_predictor = layers.LinearLayerND(
            obs_frames=self.args.obs_frames,
            pred_frames=self.args.pred_frames,
        )

        # Ego predictor
        if self.e.ego_t_f + self.e.ego_t_h > self.args.obs_frames:
            self.log('Wrong ego predictor settings (`ego_t_h` or `ego_t_f`)!',
                     level='error', raiseError=ValueError)

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
            transform=self.enc_args.T,
        )

        # Intention predictor
        if self.e.use_intention_predictor:
            self.intention_predictor = IntentionPredictor(
                obs_steps=self.args.obs_frames,
                pred_steps=self.args.pred_frames,
                ego_pred_steps=self.e.ego_t_f,
                generations=self.e.Kg,
                traj_dim=self.dim,
                feature_dim=self.args.feature_dim,
                noise_depth=self.args.noise_depth,
                transform=self.enc_args.T,
            )

        # Social predictor
        if self.e.use_social_predictor:
            self.social_predictor = SocialPredictor(
                obs_steps=self.args.obs_frames,
                pred_steps=self.args.pred_frames,
                ego_pred_steps=self.e.ego_t_f,
                partitions=self.e.partitions,
                generations=self.e.Kg,
                traj_dim=self.dim,
                feature_dim=self.args.feature_dim,
                noise_depth=self.args.noise_depth,
                transform=self.enc_args.T,
            )

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        # --------------------
        # MARK: - Preprocesses
        # --------------------
        x_ego = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        x_nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Unpacked `x_nei` are relative values, move them back
        x_nei = x_nei + x_ego[..., None, -1:, :]

        # Get types of all ego agents (if needed)
        if self.e.encode_agent_types:
            ego_types = self.get_input(inputs, INPUT_TYPES.AGENT_TYPES)
        else:
            ego_types = None

        # Other settings
        repeats = self.args.K_train if training else self.args.K

        # -------------------------
        # MARK: - Linear prediction
        # -------------------------
        if self.e.use_linear:
            y_linear = self.linear_predictor(x_ego)[..., None, :, :]
        else:
            y_linear = 0

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

        x_nei_extended = torch.concat([
            repeat(x_nei[..., None, :, :], self.e.insights, -3),
            x_s[..., 1:, :, :, :],
        ], dim=-2)

        # ---------------------------
        # MARK: - Intention predictor
        # ---------------------------
        if self.e.use_intention_predictor:
            y_intention = self.intention_predictor(
                x_ego=x_ego_extended,
                repeats=repeats,
                ego_types=ego_types,
                training=training,
            )
        else:
            y_intention = 0

        # ------------------------
        # MARK: - Social predictor
        # ------------------------
        if self.e.use_social_predictor:
            y_social = self.social_predictor(
                x_ego=x_ego_extended,
                x_nei=x_nei_extended,
                repeats=repeats,
                picker=self.picker,
                ego_types=ego_types,
                training=training,
            )
        else:
            y_social = 0

        # Final predictions
        y = y_linear + y_intention + y_social

        returns = [
            y,
        ]

        # Output predictions and labels to compute EgoLoss
        if training:
            returns += [
                x_nei[..., -_f:, :],
                yy_nei_train,
            ]

        # Visualize ego predictor's outputs
        # This only works in the playground mode
        elif v := self.e.vis_ego_predictor:
            if v == 1:
                e = torch.flatten(x_s, -4, -3)
            elif v == 2:
                yy = torch.mean(x_s, dim=-3)
                e = yy[..., 1:, :, :]
            else:
                self.log(f'Wrong `vis_ego_predictor` value recevied: {v}!',
                         level='error', raiseError=ValueError)

            returns[0] = e

        return returns


class Encore(Structure):
    MODEL_TYPE = EncoreModel

    def __init__(self, args: list[str] | Args | None = None,
                 manager: BaseManager | None = None,
                 name='Train Manager'):

        super().__init__(args, manager, name)

        ver_args = self.args.register_subargs(EncoreArgs, 'enc')

        if ver_args.ego_capacity != 0:
            self.loss.set({l2: 1.0, EgoLoss: ver_args.ego_loss_rate})
