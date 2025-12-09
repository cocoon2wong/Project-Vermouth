"""
@Author: Conghao Wong
@Date: 2025-12-09 15:50:31
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-09 15:50:33
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers


class KernelLayer(torch.nn.Module):
    """
    Kernel Layer
    ---
    The 3-layer MLP to compute reverberation kernels.
    `ReLU` is used in the first two layers, while `tanh` is used in the
    output layer.
    """

    def __init__(self, input_units: int,
                 hidden_units: int,
                 output_units: int,
                 *args, **kwargs) -> None:

        super().__init__()

        self.l1 = layers.Dense(input_units, hidden_units, torch.nn.ReLU)
        self.l2 = layers.Dense(hidden_units, hidden_units, torch.nn.ReLU)
        self.l3 = layers.Dense(hidden_units, output_units, torch.nn.Tanh)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.l3(self.l2(self.l1(f)))