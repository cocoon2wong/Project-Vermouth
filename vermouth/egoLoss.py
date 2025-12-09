"""
@Author: Conghao Wong
@Date: 2025-12-09 20:16:15
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-09 20:43:21
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.training.loss import BaseLossLayer


class EgoLoss(BaseLossLayer):

    has_unit = True

    def forward(self, outputs: list[torch.Tensor],
                labels: list[torch.Tensor],
                inputs: list[torch.Tensor],
                mask=None, training=None, *args, **kwargs):

        ego_nei_pred = outputs[2]
        ego_nei_gt = outputs[1]

        return 0
