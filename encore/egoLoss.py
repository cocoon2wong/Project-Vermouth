"""
@Author: Conghao Wong
@Date: 2025-12-09 20:16:15
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-23 11:15:59
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES
from qpid.training.loss import BaseLossLayer
from qpid.training.loss.__ade import ADE_2D
from qpid.utils import get_mask


class EgoLoss(BaseLossLayer):

    has_unit = True

    def forward(self, outputs: list[torch.Tensor],
                labels: list[torch.Tensor],
                inputs: list[torch.Tensor],
                mask=None, training=None, *args, **kwargs):

        ego_nei_pred = outputs[-1]
        ego_nei_gt = outputs[-2]

        weights = self.model.get_input(inputs, INPUT_TYPES.LOSS_WEIGHT)
        coe = self.coe * weights[:, None] if training else self.coe

        # Mask shape: (batch, nei)
        nei_mask = get_mask(torch.sum(torch.abs(ego_nei_gt), dim=[-1, -2]))

        return ADE_2D(pred=ego_nei_pred, GT=ego_nei_gt, coe=coe, mask=nei_mask)
