import pandas as pd
import sys
import numpy as np
from tslearn.metrics import dtw_path
from matplotlib.patches import ConnectionPatch
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def plot_dtw(cycle1, cycle2, global_constraint=None, savefig=False, **params_constraint):
    """
    Plots the Dynamic Time Warping (DTW) path between two cycles.

    Parameters:
    - cycle1 (array-like): First cycle data.
    - cycle2 (array-like): Second cycle data.
    - global_constraint (str): Global constraint for the DTW computation. Default is None.
    - savefig (bool): Whether to save the figure. Default is False.
    - **params_constraint: Additional parameters for the DTW computation.

    Returns:
    - None
    """
    optimal_path, dtw_score_opt = dtw_path(cycle1, cycle2, global_constraint, **params_constraint)
    
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all')
    series_line_options = {}
    axs[0].plot(cycle1, **series_line_options)
    axs[1].plot(cycle2, **series_line_options)
    
    lines = []
    warping_line_options = {'linewidth': 0.8, 'color': 'orange', 'alpha': 0.8, 'linestyle': 'dashed'}
    for r_c, c_c in optimal_path:
        if r_c < 0 or c_c < 0:
            continue
        con = ConnectionPatch(xyA=[r_c, cycle1[r_c]], coordsA=axs[0].transData,
                              xyB=[c_c, cycle2[c_c]], coordsB=axs[1].transData, **warping_line_options)
        lines.append(con)
    for line in lines:
        fig.add_artist(line)
    fig.suptitle(f'DTW optimal score: {dtw_score_opt} ({global_constraint})')
    plt.show()

def plot_dtw_mat(c1, c2, **params_constraint):
    """
    Plots the DTW cost matrix and optimal path between two cycles.

    Parameters:
    - c1 (array-like): First cycle data.
    - c2 (array-like): Second cycle data.
    - **params_constraint: Additional parameters for the DTW computation.

    Returns:
    - None
    """
    path, dtw_score_opt = dtw_path(c1, c2, **params_constraint)
    
    plt.figure(1, figsize=(8, 8))

    # Definitions for the axes
    left, bottom = 0.01, 0.1
    w_ts = h_ts = 0.2
    left_h = left + w_ts + 0.02
    width = height = 0.65
    bottom_h = bottom + height + 0.02
    sz = c1.shape[0]
    sz2 = c2.shape[0]
    rect_s_y = [left, bottom, w_ts, height]
    rect_gram = [left_h, bottom, width, height]
    rect_s_x = [left_h, bottom_h, width, h_ts]

    ax_gram = plt.axes(rect_gram)
    ax_s_x = plt.axes(rect_s_x)
    ax_s_y = plt.axes(rect_s_y)

    mat = cdist(c1.reshape(-1, 1), c2.reshape(-1, 1))

    ax_gram.imshow(mat, origin='lower', cmap='viridis')
    ax_gram.axis("on")
    ax_gram.autoscale(False)
    ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-", linewidth=3.)

    ax_s_x.plot(np.arange(sz2), c2, "b-", linewidth=2.)
    ax_s_x.axis("on")
    ax_s_x.set_xlim((0, sz2 - 1))
    ax_s_x.tick_params('both', labelbottom=False, bottom=False, labelleft=False, labelright=True, left=False, right=True)
    ax_s_y.plot(-c1, np.arange(sz), "g-", linewidth=2.)
    ax_s_y.axis("on")
    ax_s_y.set_ylim((0, sz - 1))
    ax_s_y.set_ylabel('Cycle 1')
    ax_s_x.set_title('Comparison of two cycles and optimal path', pad=20)
    ax_s_y.tick_params('both', labelbottom=False, bottom=False)
    plt.show()

def compute_dtw(data: pd.DataFrame, global_constraint="sakoe_chiba", sakoe_chiba_radius=2):
    """
    Computes the DTW distance matrix for a set of cycles.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the cycles data.
    - global_constraint (str): Global constraint for the DTW computation. Default is "sakoe_chiba".
    - sakoe_chiba_radius (int): Radius parameter for the Sakoe-Chiba global constraint. Default is 2.

    Returns:
    - DTW_matrix (ndarray): Distance matrix where DTW_matrix[i, j] is the DTW distance between cycles i and j.
    """
    N = len(data)
    DTW_matrix = np.zeros((N, N))
    
    # Compute pairwise DTW only in the upper triangular matrix
    fol = 0
    for i in range(N):
        for j in range(i, N):
            fol += 1
            if fol % 100 == 0:
                sys.stdout.write('\r')
                verbose_variable = fol / (0.5 * (N**2 + N))
                sys.stdout.write("[%-50s] %d%%" % ('=' * int(50 * verbose_variable), 100 * verbose_variable))
                sys.stdout.flush()
            _, dtw_score_opt = dtw_path(data[i], data[j], global_constraint="sakoe_chiba", sakoe_chiba_radius=4)
            DTW_matrix[i, j] = dtw_score_opt
         
    # Copy every distance on the strict lower triangular matrix
    for i in range(N):
        for j in range(i):
            DTW_matrix[i, j] = DTW_matrix[j, i]
    return DTW_matrix
