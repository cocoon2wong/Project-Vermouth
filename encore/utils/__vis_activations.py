"""
@Author: Conghao Wong
@Date: 2026-04-03 10:00:29
@LastEditors: Conghao Wong
@LastEditTime: 2026-04-03 10:01:41
@Github: https://cocoon2wong.github.io
@Copyright 2026 Conghao Wong, All Rights Reserved.
"""

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_activations(f: torch.Tensor,
                          title='Feature Activations',
                          mean_included: bool = False):
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

        if mean_included and _k == k - 1:
            s = f'Mean Rehearsal'
        else:
            s = f'Rehearsal #{_k}'

        ax.set_ylabel(f'{s}\n({percent}% Activated)')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(obs / d)

    # Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_color)
    cbar = fig.colorbar(mappable, ax=axes, location='right',
                        fraction=0.02, pad=0.02)
    cbar.set_label('Value (Color: Activated / Gray: Others)')

    plt.show()
