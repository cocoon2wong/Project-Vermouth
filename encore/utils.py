"""
@Author: Conghao Wong
@Date: 2025-12-31 11:26:02
@LastEditors: Conghao Wong
@LastEditTime: 2026-03-31 10:06:52
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch
from matplotlib import pyplot as plt


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


def visualize_insight_kernels(I: torch.Tensor, title='Insight Kernels (PCA)'):

    points = I.mean(dim=-2).cpu()

    if points.shape[-1] > 3:
        points_centered = points - points.mean(dim=-2)
        _, _, V = torch.pca_lowrank(points_centered, q=3)
        points = torch.matmul(points_centered, V[:, :3])

    points = points.numpy()

    N = points.shape[0]
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    plt.close(title)
    fig = plt.figure(title, figsize=(6.4, 4.8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(x, y, z,
                         c=z, cmap='viridis', s=60,
                         alpha=0.8, edgecolors='w',
                         linewidth=0.5)

    if N < 100:
        bbox_props = dict(
            boxstyle='round,pad=0.3',
            facecolor='gray',
            alpha=0.4,
            edgecolor='none'
        )

        for i in range(N):
            ax.text(x[i] + 0.005,
                    y[i] + 0.005,
                    z[i] + 0.005,
                    str(i),
                    color='white',
                    fontsize=9,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=bbox_props)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=15)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()
