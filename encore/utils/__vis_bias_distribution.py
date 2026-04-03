"""
@Author: Conghao Wong
@Date: 2026-04-03 10:05:02
@LastEditors: Conghao Wong
@LastEditTime: 2026-04-03 10:05:29
@Github: https://cocoon2wong.github.io
@Copyright 2026 Conghao Wong, All Rights Reserved.
"""

import torch
from matplotlib import pyplot as plt


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
