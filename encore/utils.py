"""
@Author: Conghao Wong
@Date: 2025-12-31 11:26:02
@LastEditors: Conghao Wong
@LastEditTime: 2026-04-02 20:17:52
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
import numpy as np
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


def visualize_insight_kernels(I: torch.Tensor,
                              IDs: list[str],
                              title='Insight Kernels (PCA)'):
    """
    Visualization of the feature distribution of time-averaged insight kernels
    in the 3D space. For insight kernels with more than 3 number of insights,
    i.e., $K_I > 3$, PCA dimension reduction will be performed automatically.
    """

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

    if N < 50:
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

    else:
        import mplcursors

        cursor = mplcursors.cursor(scatter, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            label = IDs[sel.index]
            sel.annotation.set_text(label)
            sel.annotation.get_bbox_patch().set(fc="gray", alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=15)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def visualize_max_activations(f: torch.Tensor,
                              title='Feature Activations'):
    """
    Visualize all selected features after applying the feature-level
    conditioning (dimension-wise max-pooling) on the feature of
    biased observations.
    Shape of the input feature `f` should be `(K_I, obs, d)`.
    """
    k, obs, d = f.shape[-3:]

    v_max = f.max().item()
    v_min = f.min().item()

    # Indices -> (obs, d)
    activate_idx = torch.argmax(f, dim=0)

    # Normalize
    norm = colors.Normalize(vmin=v_min, vmax=v_max)
    norm_data = norm(f.cpu().numpy())

    # Activated: RGB colors
    # Deactivated: Grey colors
    cmap_color = cm.get_cmap('viridis')
    rgb_color = cmap_color(norm_data)[..., :3]

    # -> (KI, obs, d, 3)
    gray_val = norm_data    # (k, obs, d)
    rgb_gray = np.stack([gray_val, gray_val, gray_val], axis=-1)

    # Color activated parts
    final_rgb = rgb_gray.copy()

    for _k in range(k):
        mask = (activate_idx == _k).cpu().numpy()
        final_rgb[_k][mask] = rgb_color[_k][mask]

    # Start visualizing
    plt.close(title)

    fig, axes = plt.subplots(k, 1, figsize=(d * 0.25, k * obs * 0.3),
                             sharex=True, constrained_layout=True, num=title)
    if k == 1:
        axes = [axes]

    for _k in range(k):
        ax = axes[_k]

        ax.imshow(final_rgb[_k], aspect='auto',
                  interpolation='nearest')

        for i in range(obs):
            for j in range(d):
                if activate_idx[i, j] == _k:
                    rect = patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        linewidth=1.0,
                        edgecolor='cyan',
                        facecolor='none'
                    )
                    ax.add_patch(rect)

        # Compute the activation rate
        count = (activate_idx == _k).sum().item()
        percent = int((count / (obs * d)) * 1000)/10

        ax.set_ylabel(f'Rehearsal #{_k}\n({percent}% Activated)')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(obs / d)

    # Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_color)
    cbar = fig.colorbar(mappable, ax=axes, location='right',
                        fraction=0.02, pad=0.02)
    cbar.set_label('Value (Color: Activated / Gray: Others)')

    plt.show()
