import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

CMAP = matplotlib.cm.get_cmap('cividis')
HIST_COLOR = CMAP(0.6)


def plot_hist2d(x, y, ma, step, out_dir=None, figsize=(8, 6), xlabel='Ground truth [m]', ylabel='Prediction [m]',
                usetex=False, fontsize=18, bins=None, vmax=None, cmap='cividis'):
    """
    Confusion plot ground truth values vs. predicted values.
    """
    if bins is None:
        edges = np.linspace(0, ma, 100)
        bins = (edges, edges)

    fig = plt.figure(figsize=figsize)
    h = plt.hist2d(x.squeeze(), y.squeeze(), bins=bins, cmap=cmap,
                   norm=matplotlib.colors.LogNorm(), vmax=vmax, rasterized=True)
    plt.colorbar(h[3], label='Number of samples')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.axis('equal')
    plt.plot([0, ma], [0, ma], 'k--')
    plt.xlim((0, ma))
    plt.ylim((0, ma))
    ticks = np.arange(0, ma + step, step)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.tight_layout()
    if out_dir:
        fig.savefig(fname=os.path.join(out_dir, 'confusion.png'), dpi=300)
    return fig


def plot_precision_recall(predictions, targets, uncertainties, metric='RMSE',
                          ax=None, figsize=(8, 6), ylabel='RMSE', label=None, style=None, out_dir=None):
    # compute errors
    errors = predictions - targets

    # sort data
    sorted_inds = uncertainties.argsort()
    uncertainties = uncertainties[sorted_inds]
    errors = errors[sorted_inds]

    precision, recall = [], []

    num_total = len(errors)
    for i in range(num_total):
        if metric == 'RMSE':
            precision.append(np.sqrt(np.mean(errors[0:i] ** 2)))
        elif metric == 'MSE':
            precision.append(np.mean(errors[0:i] ** 2))

        recall.append(i / num_total)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    if style is not None:
        ax.plot(recall, precision, style, label=label)
    else:
        ax.plot(recall, precision, label=label)

    ax.set_xlabel('Recall')
    ax.set_ylabel(ylabel)

    if out_dir:
        fig.savefig(fname=os.path.join(out_dir, 'precision_recall_curve.png'), dpi=300)

    return ax


def plot_calibration(predictions, targets, uncertainties, min_bin_count=10, metric='RMSE', bins=None, step=None,
                     style='k-o', ax=None, figsize=(8, 6), xlabel='STD', ylabel='RMSE', out_dir=None):
    """
    uncertainties: must be standard deviations -> will be squared for MSE.
    """
    color_ax = CMAP(0.05)
    color_ax2 = HIST_COLOR

    if bins is None:
        ma = np.max(uncertainties)
        step = ma / 10
        bins = np.arange(0, ma + step, step)
        print(bins)
    else:
        ma = np.max(bins)

    # bin data
    bin_indices = np.digitize(x=uncertainties, bins=bins, right=True)

    errors = predictions - targets

    error_binned, uncertainty_binned, num_binned = [], [], []

    for idx in np.arange(len(bins)) + 1:
        bin_i_indices = bin_indices == idx

        if metric == 'RMSE':
            # average the estimated uncertainty per bin (var or std)
            uncertainty_binned.append(np.sqrt(np.mean(uncertainties[bin_i_indices] ** 2)))
            # average the respective error metric per bin
            error_binned.append(np.sqrt(np.mean(errors[bin_i_indices] ** 2)))

        elif metric == 'MSE':
            # average the estimated uncertainty per bin (var or std)
            uncertainty_binned.append(np.mean(uncertainties[bin_i_indices] ** 2))
            # average the respective error metric per bin
            error_binned.append(np.mean(errors[bin_i_indices] ** 2))

        num_binned.append(np.sum(bin_i_indices))

    # convert to numpy
    error_binned = np.array(error_binned)
    uncertainty_binned = np.array(uncertainty_binned)
    num_binned = np.array(num_binned)

    # remove estimates where the number of samples per bin is to small
    error_binned[num_binned < min_bin_count] = np.nan
    uncertainty_binned[num_binned < min_bin_count] = np.nan

    if ax is None:
        # fig, ax = plt.subplots(figsize=figsize)
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    ax.plot(uncertainty_binned, error_binned, style, zorder=2, color=color_ax)
    print('x: ', uncertainty_binned)
    print('y: ', error_binned)
    print('count: ', num_binned)

    ax.set_xlim(0, ma)
    ax.set_ylim(0, ma)

    # perfect calibration line
    ax.plot(ax.get_xlim(), ax.get_xlim(), "k--", zorder=1, alpha=0.5)

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.bar(bins, num_binned, align='edge', width=step, zorder=0, edgecolor='white', color=color_ax2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, color=color_ax)
    ax2.set_ylabel('Number of samples', color=color_ax2)
    ax2.tick_params(axis='y', labelcolor=color_ax2)

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.tick_params(axis='y', labelcolor=color_ax)

    # ax.set_aspect("equal")
    fig.tight_layout()

    if out_dir:
        fig.savefig(fname=os.path.join(out_dir, 'calibration.png'), dpi=300)

    return fig, ax



