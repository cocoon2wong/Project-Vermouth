"""
@Author: Conghao Wong
@Date: 2025-12-02 11:10:53
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-25 10:59:22
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
from .egoPredictor import EgoPredictor
from .intentionPredictor import IntentionPredictor
from .socialPredictor import SocialPredictor


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

        # Set model inputs and labels
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.NEIGHBOR_TRAJ)

        # Layers
        # Transform layers
        t_type, it_type = layers.get_transform_layers(self.enc_args.T)
        self.tlayer = t_type((self.args.obs_frames, self.dim))
        self.itlayer = it_type((self.args.pred_frames, self.dim))

        # Ego predictor
        if self.e.ego_t_f + self.e.ego_t_h > self.args.obs_frames:
            self.log('Wrong ego predictor settings (`ego_t_h` or `ego_t_f`)!',
                     level='error', raiseError=ValueError)

        self.ego_predictor = EgoPredictor(
            obs_steps=self.e.ego_t_h,
            pred_steps=self.e.ego_t_f,
            insights=self.e.insights,
            capacity=self.e.ego_capacity,
            traj_dim=self.dim,
            feature_dim=self.args.feature_dim//2,
            noise_depth=self.args.noise_depth,
            transform=self.enc_args.T,
        )

        self.intention_predictor = IntentionPredictor(
            obs_steps=self.args.obs_frames,
            pred_steps=self.args.pred_frames,
            generations=self.e.Kg,
            traj_dim=self.dim,
            feature_dim=self.args.feature_dim,
            noise_depth=self.args.noise_depth,
            transform=self.enc_args.T,
        )

        self.social_predictor = SocialPredictor(
            obs_steps=self.args.obs_frames,
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

        # Unpacked `x_nei` are relative values, move them back
        x_nei = x_nei + x_ego[..., None, -1:, :]

        repeats = self.args.K_train if training else self.args.K

        # ---------------------------
        # MARK: - Intention predictor
        # ---------------------------
        y_intention, y_linear = self.intention_predictor(
            x_ego=x_ego,
            repeats=repeats,
            training=training,
        )

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
        yy_nei_original = self.ego_predictor(
            x_ego=x_ego[..., -_h:, :],
            x_nei=torch.concat([x_ego[..., None, -_h:, :],
                                x_nei[..., -_h:, :]], dim=-3),
            training=training,
        )  # -> (batch, nei+1, insights, ego_t_f, dim)

        # -> (batch, nei+1, ego_t_f, dim)
        yy = torch.mean(yy_nei_original, dim=-3)
        yy_ego = yy[..., 0, :, :]
        yy_nei = yy[..., 1:, :, :]

        # ------------------------
        # MARK: - Social predictor
        # ------------------------
        # "Mess Up" the time axis
        x_ego_old = x_ego
        x_nei_old = x_nei

        x_ego = torch.concat([x_ego[..., -_h:, :], yy_ego], dim=-2)
        x_nei = torch.concat([x_nei[..., -_h:, :], yy_nei], dim=-2)

        y_social = self.social_predictor(
            x_ego=x_ego,
            x_nei=x_nei,
            repeats=repeats,
            picker=self.picker,
            training=training,
        )

        # Final predictions
        y = y_linear[..., None, :, :] + y_intention + y_social

        returns = [
            y,
        ]

        # Output predictions and labels to compute EgoLoss
        if training:
            returns += [
                x_nei_old[..., -_f:, :],
                yy_nei_train,
            ]

        # Visualize ego predictor's outputs
        # This only works in the playground mode
        elif v := self.e.vis_ego_predictor:
            if v == 1:
                e = torch.flatten(yy_nei_original, -4, -3)
            elif v == 2:
                e = yy_nei
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

        self.ver_args = self.args.register_subargs(EncoreArgs, 'enc')

        self.loss.set({l2: 1.0, EgoLoss: self.ver_args.ego_loss_rate})
