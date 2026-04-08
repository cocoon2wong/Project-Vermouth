"""
@Author: Conghao Wong
@Date: 2026-04-03 10:00:29
@LastEditors: Conghao Wong
@LastEditTime: 2026-04-08 09:29:10
@Github: https://cocoon2wong.github.io
@Copyright 2026 Conghao Wong, All Rights Reserved.
"""

import os

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def vis_activations(f: torch.Tensor,
                    title='Feature Activations',
                    last_row_name: str | None = None,
                    deviation_mode: bool = False):
    """
    Visualize all selected features after applying the feature-level
    conditioning (dimension-wise max-pooling) on the feature of
    biased observations.
    Shape of the input feature `f` should be `(K_I, obs, d)`.
    """
    k, obs, d = f.shape[-3:]

    # Colors for deactivated features
    rng = np.random.default_rng(seed=42)

    # RGB -> BGR
    c = 0.6 + 0.4 * rng.random((k, 3))
    c = np.column_stack([c.T[2], c.T[1], c.T[0]])

    # Visualize feature deviations
    if deviation_mode:
        f = torch.abs(f - f.mean(dim=0, keepdim=True))
        mode_str = 'Deviated'
        label_str = 'Abs Deviation'

        if 'Activations' in title:
            title = title.replace('Activations', 'Deviations')

    # Visualize feature activations (maxpool)
    else:
        mode_str = 'Activated'
        label_str = 'Value'

    v_max = f.max().item()
    v_min = f.min().item()

    # Indices -> (obs, d)
    activate_idx = torch.argmax(f, dim=0)

    # Normalize
    norm = colors.Normalize(vmin=v_min, vmax=v_max)
    norm_data = norm(f.cpu().numpy())

    # Activated: RGB colors
    # Deactivated: Random gradient colors (per matrix)
    cmap_color = cm.get_cmap('viridis')
    rgb_color = cmap_color(norm_data)[..., :3]

    # -> (KI, obs, d, 3)
    final_rgb = np.zeros((k, obs, d, 3))

    for _k in range(k):
        final_rgb[_k] = 0.3 + 0.7 * norm_data[_k, ..., None] * c[_k]
        mask = (activate_idx == _k).cpu().numpy()
        final_rgb[_k][mask] = rgb_color[_k][mask]

    # Start visualizing
    plt.close(title)
    axes: list[Axes]

    fig, axes = plt.subplots(
        nrows=k, ncols=1,
        figsize=(d * 0.25, k * obs * 0.3),
        sharex=True, constrained_layout=True, num=title
    )

    if k == 1:
        axes = [axes]  # type: ignore

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
                        facecolor='none',
                        clip_on=False,
                    )
                    ax.add_patch(rect)

        # Compute the activation rate
        count = (activate_idx == _k).sum().item()
        percent = int((count / (obs * d)) * 1000)/10

        if last_row_name is not None and _k == k - 1:
            s = last_row_name
        else:
            s = f'Rehearsal #{_k}'

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)

        ax.set_ylabel(f'{s}\n({percent}% {mode_str})')
        ax.set_box_aspect(obs / d)

    # Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_color)
    cbar = fig.colorbar(mappable, ax=axes, location='right',
                        fraction=0.02, pad=0.02)

    cbar.set_label(
        f'{label_str} (Color: {mode_str} / Background: {c.shape[0]} Colors)')

    # save_as_subfigures(fig, axes, title)
    plt.show()


def save_as_subfigures(fig: Figure,
                       axes: list[Axes],
                       base_name: str,
                       path: str = './'):

    fig.canvas.draw()

    for _k, ax in enumerate(axes):
        bbox = ax.get_tightbbox(fig.canvas.get_renderer())

        if bbox is None:
            continue

        extent = bbox.transformed(fig.dpi_scale_trans.inverted())
        extent = extent.padded(0.08)

        save_path = os.path.join(path, f'{base_name}_rehearsal{_k}.png')
        fig.savefig(save_path, bbox_inches=extent, dpi=300)
