"""
@Author: Conghao Wong
@Date: 2025-12-02 11:10:53
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-09 19:40:56
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model
from qpid.training import Structure

from .egoPredictor import EgoPredictor


class VermouthModel(Model):
    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.NEIGHBOR_TRAJ)

        self.ego_predictor = EgoPredictor(
            obs_steps=self.args.obs_frames//2,
            pred_steps=self.args.obs_frames//2,
            insights=5,
            traj_dim=self.dim,
            feature_dim=self.args.feature_dim,
        )

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        obs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        ego_nei_pred = self.ego_predictor(
            ego_traj=obs[..., :self.args.obs_frames//2, :],
            nei_trajs=nei[..., :self.args.obs_frames//2, :]
        )

        return torch.repeat_interleave(
            obs[..., -1:, :],
            repeats=self.args.pred_frames,
            dim=-2)


class Vermouth(Structure):
    MODEL_TYPE = VermouthModel
    is_trainable = False
