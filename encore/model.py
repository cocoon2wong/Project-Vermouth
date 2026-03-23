"""
@Author: Conghao Wong
@Date: 2025-12-02 11:10:53
@LastEditors: Conghao Wong
@LastEditTime: 2026-03-23 19:18:29
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
    The *Encore* trajectory prediction model. It consists of two main
    parts: the Ego Predictor and the Final Predictor. The Ego Predictor
    provides diverse biased short-term future predictions (rehearsals)
    for all neighboring agents from each ego agent's unique point of
    view. The Final Predictor then utilizes these biased observations
    to achieve conditioned final predictions.
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

        # Set model inputs.
        # Types of agents are only used in complex scenes.
        # For other datasets, keep it disabled (through the arg).
        if not self.e.encode_agent_types:
            self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                            INPUT_TYPES.NEIGHBOR_TRAJ)
        else:
            self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                            INPUT_TYPES.NEIGHBOR_TRAJ,
                            INPUT_TYPES.AGENT_TYPES)

        # ---------------------
        # MARK: - Ego predictor
        # ---------------------
        # Length check
        if self.e.ego_t_f + self.e.ego_t_h > self.args.obs_frames:
            self.log('Wrong ego predictor settings (`ego_t_h` or `ego_t_f`)!',
                     level='error', raiseError=ValueError)

        # `ego_capacity = 0` is only used in ablations.
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
            feature_dim=self.args.feature_dim // 2,
            noise_depth=self.args.noise_depth,
            transform=self.e.T,
            compute_ego_bias=self.e.compute_ego_bias,
        )

        # -----------------------
        # MARK: - Final predictor
        # -----------------------
        # The following networks come from "Reverberation: Learning the
        # Latencies Before Forecasting Trajectories".

        # Linear predictor
        self.linear_predictor = layers.LinearLayerND(
            obs_frames=self.args.obs_frames + self.e.ego_t_f,
            pred_frames=self.args.pred_frames - self.e.ego_t_f,
            return_full_trajectory=True,
        )

        # Intention predictor
        if self.e.use_intention_predictor:
            self.intention_predictor = IntentionPredictor(
                obs_steps=self.args.obs_frames + self.e.ego_t_f,
                pred_steps=self.args.pred_frames,
                generations=self.e.Kg,
                traj_dim=self.dim,
                feature_dim=self.args.feature_dim,
                noise_depth=self.args.noise_depth,
                transform=self.enc_args.T,
            )

        # Social predictor
        if self.e.use_social_predictor:
            self.social_predictor = SocialPredictor(
                obs_steps=self.args.obs_frames + self.e.ego_t_f,
                pred_steps=self.args.pred_frames,
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

        # The unpacked `x_nei` values are relative; convert them back to
        # absolute coordinates.
        x_nei = x_nei + x_ego[..., None, -1:, :]

        # Get the types of all ego agents (if needed).
        if self.e.encode_agent_types:
            ego_types = self.get_input(inputs, INPUT_TYPES.AGENT_TYPES)
        else:
            ego_types = None

        # Other settings
        repeats = self.args.K_train if training else self.args.K

        # ---------------------
        # MARK: - Ego predictor
        # ---------------------
        _h = self.enc_args.ego_t_h
        _f = self.enc_args.ego_t_f

        # Training of the ego predictor: on observation steps only.
        if training:
            yy_nei_train = self.ego_predictor(
                x_ego=x_ego[..., -(_h + _f):-_f, :],
                x_nei=x_nei[..., -(_h + _f):-_f, :],
                training=training,
            )  # -> (batch, nei, insights, ego_t_f, dim)

        # Normal use of the ego predictor.
        # Also predict the ego agent's trajectory.
        x_s = self.ego_predictor(
            x_ego=x_ego[..., -_h:, :],
            x_nei=torch.concat([x_ego[..., None, -_h:, :],
                                x_nei[..., -_h:, :]], dim=-3),
            training=training,
        )  # -> (batch, nei+1, insights, ego_t_f, dim)

        # Unpack the ego predictor's predictions and concatenate the
        # observations with the short-term predictions.
        x_ego_extended = torch.concat([
            repeat(x_ego[..., None, :, :], self.e.insights, -3),
            x_s[..., 0, :, :, :]
        ], dim=-2)

        x_nei_extended = torch.concat([
            repeat(x_nei[..., None, :, :], self.e.insights, -3),
            x_s[..., 1:, :, :, :],
        ], dim=-2)

        # -----------------------
        # MARK: - Final predictor
        # -----------------------
        # -------------------------
        # MARK: - Linear prediction
        # -------------------------
        if self.e.use_linear:
            y_linear = self.linear_predictor(
                torch.mean(x_ego_extended, dim=-3)
            )[..., None, -self.args.pred_frames:, :]
        else:
            y_linear = 0

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

        # Output predictions and labels to compute the EgoLoss.
        if training:
            returns += [
                x_nei[..., -_f:, :],
                yy_nei_train,
            ]

        # Visualize the ego predictor's outputs.
        # This only works in the playground mode.
        elif v := self.e.vis_ego_predictor:
            if v == 1:
                e = x_s
            elif v == 2:
                yy = torch.mean(x_s, dim=-3)
                e = yy[..., 1:, :, :]
            else:
                self.log(f'Wrong `vis_ego_predictor` value received: {v}!',
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
