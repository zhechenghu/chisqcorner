
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from scipy.stats import binned_statistic, binned_statistic_2d
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


def plot_corner_chi2(samples_df: pd.DataFrame,
                     param_labels: list[str],
                     colormap: Colormap,
                     nbins: int,
                     chi2_col: str = 'chi2') -> Figure:
    """
    Generates a corner plot showing minimum chi^2 values for pairs of parameters
    and 1D marginalized chi^2 profiles.

    Args:
        samples_df: DataFrame containing parameter values and a chi^2 column.
                      Assumed to be cleaned (no NaNs or Infs in relevant columns).
        param_labels: List of labels for the parameters, in the same order as
                      parameter columns will be derived from samples_df.
        colormap: Matplotlib colormap for the 2D plots.
        nbins: Number of bins for histograms and 2D grids.
        chi2_col: Name of the chi-squared column in samples_df.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    params = [col for col in samples_df.columns if col != chi2_col]
    n_params = len(params)

    if len(params) != len(param_labels):
        raise ValueError(
            f"Number of parameters derived from DataFrame ({len(params)}) "
            f"does not match number of labels provided ({len(param_labels)})."
        )

    # --- Pre-calculate statistics and ranges ---

    # 1. For 2D plots (lower triangle)
    all_2d_stats_data = {}  # To store (min_chi2_2d, x_edges, y_edges)
    all_min_chi2_values_for_2d_range = []

    for i_row in range(n_params):
        for j_col in range(n_params):
            if i_row > j_col:  # Lower triangle
                param_x_name = params[j_col]
                param_y_name = params[i_row]

                x_data = samples_df[param_x_name]
                y_data = samples_df[param_y_name]
                chi2_values = samples_df[chi2_col]

                if x_data.empty or y_data.empty:
                    all_2d_stats_data[(i_row, j_col)] = (np.full((nbins, nbins), np.nan),
                                                       np.array([0, 1]), np.array([0, 1]))
                    continue

                bins_x = np.linspace(x_data.min(), x_data.max(), nbins + 1)
                bins_y = np.linspace(y_data.min(), y_data.max(), nbins + 1)

                min_chi2_2d_stat, x_edges_stat, y_edges_stat, _ = binned_statistic_2d(
                    x_data, y_data, chi2_values,
                    statistic='min', bins=[bins_x, bins_y]
                )
                all_2d_stats_data[(i_row, j_col)] = (min_chi2_2d_stat, x_edges_stat, y_edges_stat)
                if not np.all(np.isnan(min_chi2_2d_stat)):
                    all_min_chi2_values_for_2d_range.append(min_chi2_2d_stat[~np.isnan(min_chi2_2d_stat)])

    concatenated_2d_chi2s = np.concatenate(all_min_chi2_values_for_2d_range) if all_min_chi2_values_for_2d_range else np.array([])

    if concatenated_2d_chi2s.size > 0 and np.any(np.isfinite(concatenated_2d_chi2s)):
        reference_chi2_for_delta_2d = np.nanmin(concatenated_2d_chi2s)
        global_max_chi2_2d = np.nanmax(concatenated_2d_chi2s)
        if reference_chi2_for_delta_2d == global_max_chi2_2d:
            delta_adjust = max(0.1, abs(reference_chi2_for_delta_2d * 0.1)) if reference_chi2_for_delta_2d != 0 else 0.1
            global_max_chi2_2d += delta_adjust
    else:
        reference_chi2_for_delta_2d = 0.0
        global_max_chi2_2d = 1.0

    overall_final_vmax_delta = global_max_chi2_2d - reference_chi2_for_delta_2d
    if overall_final_vmax_delta <= 0.0:
        overall_final_vmax_delta = 0.1

    significant_chi2_levels = np.array([0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    boundaries_list = [0.0]
    for level in significant_chi2_levels[1:]:
        if level < overall_final_vmax_delta:
            boundaries_list.append(level)
        else:
            break
    boundaries_list.append(overall_final_vmax_delta)
    cmap_boundaries = np.unique(np.array(boundaries_list))

    if len(cmap_boundaries) < 2 and overall_final_vmax_delta > 0:
        cmap_boundaries = np.array([0.0, overall_final_vmax_delta])
    elif len(cmap_boundaries) < 2:
        cmap_boundaries = np.array([0.0, 0.1])

    custom_norm = BoundaryNorm(cmap_boundaries, ncolors=colormap.N, clip=True)

    # 2. For 1D plots (diagonal)
    all_1d_stats_data = {}
    all_min_chi2_values_for_1d_range = []

    for k_diag in range(n_params):
        param_name = params[k_diag]
        param_values = samples_df[param_name]
        chi2_values = samples_df[chi2_col]

        if param_values.empty:
            all_1d_stats_data[k_diag] = (np.full(nbins, np.nan),
                                       np.linspace(0, 1, nbins),
                                       np.linspace(0, 1, nbins + 1))
            continue

        bin_edges_1d = np.linspace(param_values.min(), param_values.max(), nbins + 1)
        min_chi2_1d_stat, _, _ = binned_statistic(
            param_values, chi2_values, statistic='min', bins=bin_edges_1d
        )
        bin_centers_1d = (bin_edges_1d[:-1] + bin_edges_1d[1:]) / 2
        all_1d_stats_data[k_diag] = (min_chi2_1d_stat, bin_centers_1d, bin_edges_1d)
        if not np.all(np.isnan(min_chi2_1d_stat)):
            all_min_chi2_values_for_1d_range.append(min_chi2_1d_stat[~np.isnan(min_chi2_1d_stat)])

    concatenated_1d_chi2s = np.concatenate(all_min_chi2_values_for_1d_range) if all_min_chi2_values_for_1d_range else np.array([])
    if concatenated_1d_chi2s.size > 0 and np.any(np.isfinite(concatenated_1d_chi2s)):
        global_min_chi2_1d = np.nanmin(concatenated_1d_chi2s)
        global_max_chi2_1d = np.nanmax(concatenated_1d_chi2s)
        if global_min_chi2_1d == global_max_chi2_1d:
            delta = max(0.1, abs(global_min_chi2_1d * 0.1)) if global_min_chi2_1d != 0 else 0.1
            global_min_chi2_1d -= delta
            global_max_chi2_1d += delta
    else:
        global_min_chi2_1d = 0.0
        global_max_chi2_1d = 1.0

    # --- Plotting ---
    fig_size = max(8.0, n_params * 2.5)
    fig, axes = plt.subplots(n_params, n_params, figsize=(fig_size+2, fig_size))
    if n_params == 1:
        axes = np.array([[axes]])
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    pcm_for_colorbar = None

    for i_row in range(n_params):
        for j_col in range(n_params):
            ax = axes[i_row, j_col]

            if i_row < j_col:  # Upper triangle: hide
                ax.set_visible(False)
                continue

            current_param_x_name = params[j_col]
            # current_param_y_name = params[i_row] # Not explicitly used by name below

            if i_row == j_col:  # Diagonal plots
                min_chi2_1d, bin_centers, _ = all_1d_stats_data[i_row]
                delta_chi2_1d = min_chi2_1d - global_min_chi2_1d
                ax.plot(bin_centers, delta_chi2_1d, drawstyle='steps-mid', color='dodgerblue')
                ax.set_ylim(0, global_max_chi2_1d - global_min_chi2_1d)
                if samples_df[current_param_x_name].notna().any():
                    ax.set_xlim(samples_df[current_param_x_name].min(), samples_df[current_param_x_name].max())
                else:
                    ax.set_xlim(0, 1)

            elif i_row > j_col:  # Lower triangle: 2D chi2 maps
                min_chi2_2d, x_edges, y_edges = all_2d_stats_data[(i_row, j_col)]
                delta_chi2_map = min_chi2_2d - reference_chi2_for_delta_2d
                
                final_vmin_delta = 0.0
                # overall_final_vmax_delta is calculated before norm setup
                # final_vmax_delta = overall_final_vmax_delta # Use this pre-calculated value

                pcm = ax.pcolormesh(x_edges, y_edges, delta_chi2_map.T,
                                    cmap=colormap, norm=custom_norm,
                                    shading='auto')
                if pcm_for_colorbar is None: pcm_for_colorbar = pcm

                if not np.all(np.isnan(delta_chi2_map)):
                    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                    contour_levels = np.array([n**2 for n in range(1, 11)])
                    
                    # Use overall_final_vmax_delta for contour bounds
                    valid_levels = contour_levels[(contour_levels > final_vmin_delta) & (contour_levels < overall_final_vmax_delta)]

                    if valid_levels.size > 0:
                        ax.contour(x_centers, y_centers, delta_chi2_map.T,
                                   levels=valid_levels, colors='white', linewidths=0.75, alpha=0.6)

                ax.set_xlim(x_edges.min(), x_edges.max())
                ax.set_ylim(y_edges.min(), y_edges.max())

            # Axis labels
            if i_row == n_params - 1:
                ax.set_xlabel(param_labels[j_col])
            
            if j_col == 0:
                if i_row == 0:
                    ax.set_ylabel(r'$\Delta \chi^2$')
                elif i_row != 0:
                    ax.set_ylabel(param_labels[i_row])
            
            if i_row < n_params - 1:
                ax.set_xticklabels([])
            if j_col > 0 and i_row != j_col:  # Don't remove y tick labels for diagonal plots
                ax.set_yticklabels([])
            
            # For diagonal plots, add tick labels on the right side
            if i_row == j_col:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position('right')
            
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune='both'))
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune='both'))

    if n_params > 1 and pcm_for_colorbar is not None:
        fig.colorbar(pcm_for_colorbar, ax=axes.ravel().tolist(), label=r'$\Delta \chi^2$',
                     aspect=40, pad=0.03, shrink=0.8, ticks=cmap_boundaries)

    #fig.suptitle('Corner Plot: Minimum $\chi^2$ per Bin', fontsize=16)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig