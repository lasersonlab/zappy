import numpy as np
import pandas as pd

import matplotlib.patches as mpatches
from matplotlib import pylab
import seaborn as sns

import util


def plot_jobs(normalized_times_df, point_size=4, plot_step=100, ax=None):
    """
    Plot basic history of job runtimes
    """

    palette = sns.color_palette("hls", len(util.DEFAULT_FIELDS))
    create_fig = ax is None

    if create_fig:
        fig = pylab.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
    total_jobs = len(normalized_times_df)

    y = np.arange(len(normalized_times_df))
    point_size = 8

    patches = []
    for f_i, field_name in enumerate(util.DEFAULT_FIELDS):
        ax.scatter(
            normalized_times_df[field_name],
            y,
            c=palette[f_i],
            edgecolor="none",
            s=point_size,
            alpha=0.8,
        )

        patches.append(mpatches.Patch(color=palette[f_i], label=field_name))
    ax.set_xlabel("wallclock time (sec)")
    ax.set_ylabel("job")

    legend = pylab.legend(handles=patches, loc="upper right", frameon=True)
    legend.get_frame().set_facecolor("#FFFFFF")

    ax.xaxis.set_major_locator(pylab.MaxNLocator(min_n_ticks=10))

    plot_step = 100  # int(np.min([128, total_jobs/32]))
    y_ticks = np.arange(total_jobs // plot_step + 2) * plot_step
    ax.set_yticks(y_ticks)
    ax.set_ylim(-0.02 * total_jobs, total_jobs * 1.05)

    ax.set_xlim(-5, np.max(normalized_times_df["results returned"]) * 1.05)
    for y in y_ticks:
        ax.axhline(y, c="k", alpha=0.1, linewidth=1)

    ax.grid(False)
    if create_fig:
        fig.tight_layout()
    return ax
