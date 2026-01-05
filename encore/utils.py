"""
@Author: Conghao Wong
@Date: 2025-12-31 11:26:02
@LastEditors: Conghao Wong
@LastEditTime: 2026-01-05 19:18:26
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


def tensor_size_mb(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / 1024**2


def print_variable_summary(locals):
    items = []
    for k, v in locals.items():
        if torch.is_tensor(v):
            items.append((tensor_size_mb(v), k, v))

    items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
    for size, k, v in items_sorted:
        print(
            f"{k:>25}  {str(tuple(v.shape)):>20}  {size:10.2f} MB  {v.dtype}  {v.device}")

    # summary
    total = sum(s for s, _, _ in items_sorted)
    n = len(items_sorted)
    max_item = items_sorted[0] if n else None

    print("-" * 100)
    if n and max_item:
        print(
            f"SUMMARY: {n} tensors, total {total:.2f} MB, largest {max_item[1]} {tuple(max_item[2].shape)} {max_item[0]:.2f} MB")
    else:
        print("SUMMARY: 0 tensors")
