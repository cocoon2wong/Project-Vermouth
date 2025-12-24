"""
@Author: Conghao Wong
@Date: 2025-12-23 11:10:36
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-23 16:14:04
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


class ReverberationTransform(torch.nn.Module):
    """
    Reverberation Transform Layer
    ---
    The reverberation transform layer, which applies the proposed reverberation
    transform on the given sequential representation.

    NOTE: This layer does not contain the trainable reverberation kernels. Please
    train them outside from this class (layer).
    """

    def __init__(self, historical_steps: int,
                 future_steps: int,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.T_h = historical_steps
        self.T_f = future_steps

        self.outer = layers.OuterLayer(self.T_h, self.T_h)

    def forward(self, f: torch.Tensor,
                R: torch.Tensor | torch.nn.Module,
                G: torch.Tensor | torch.nn.Module) -> torch.Tensor:
        """
        The reverberation transform.
        `f` is the representation of a sequential input, with a shape of
        `(..., steps, d)`.

        If any of `R` or `G` is an instance of `torch.nn.Module`, it will
        directly apply that layer on the input `f`, without applying the
        original reverberation transform.
        When `R` or `G` are `torch.nn.Module`s, their input-output shapes
        should satisfy:
        - `G`: `(..., d, T_h, T_h)` -> `(..., d, K_g, T_h)`;
        - `R`: `(..., d, K_g, T_h)` -> `(..., K_g, T_f, d)`.
        """

        # Outer product
        f_t = torch.transpose(f, -1, -2)            # (..., d, T_h)
        f_o = self.outer(f_t, f_t)                  # (..., d, T_h, T_h)

        # Apply the generating kernel
        if isinstance(G, torch.Tensor):
            f = f_o @ G[..., None, :, :]            # (batch, d, T_h, K_g)
            f = torch.transpose(f, -1, -2)          # (batch, d, K_g, T_h)
        elif isinstance(G, torch.nn.Module):
            f = G(f_o)
        else:
            raise ValueError('Illegal value received (Generating Kernel)!')

        if isinstance(R, torch.Tensor):
            # `f` should now has the shape `(batch, d, K_g, T_h)`
            f = f @ R[..., None, :, :]              # (batch, d, K_g, T_f)
            f = torch.transpose(f, -1, -3)          # (batch, T_f, K_g, d)
            f = torch.transpose(f, -2, -3)          # (batch, K_g, T_f, d)
        elif isinstance(R, torch.nn.Module):
            f = R(f)
        else:
            raise ValueError('Illegal value received (Reverberation Kernel)!')

        return f
