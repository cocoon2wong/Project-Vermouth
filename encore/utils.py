"""
@Author: Conghao Wong
@Date: 2025-12-31 11:26:02
@LastEditors: Conghao Wong
@LastEditTime: 2025-12-31 14:20:11
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch


def repeat(input: torch.Tensor, repeats: int, dim: int):
    shape = input.shape
    d = dim % len(shape)
    x = input.unsqueeze(d+1)
    x = x.expand(*shape[:d+1], repeats, *shape[d+1:])
    x = x.flatten(d, d+1)
    return x
