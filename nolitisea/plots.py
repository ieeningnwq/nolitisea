import sys
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
#  Publication-style configuration
# ============================================================

def _get_serif_fonts():
    """Return a list of serif fonts available across platforms."""
    if sys.platform == "win32":
        return ["Times New Roman", "DejaVu Serif", "serif"]
    elif sys.platform == "darwin":
        return ["Times", "Times New Roman", "DejaVu Serif", "serif"]
    else:
        return ["DejaVu Serif", "Liberation Serif", "serif"]


def set_publication_style():
    """Set matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": _get_serif_fonts(),
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        # Axes
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.top": True,
        "ytick.right": True,
        # Figure
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.5",
        "legend.fancybox": False,
    })


# ============================================================
#  Plotting functions
# ============================================================

def plot_clean_vs_noisy(
    clean,
    noisy,
    *,
    sample=None,
    xlabel="Sample index $n$",
    ylabel="Signal value",
    title=None,
    figsize=(7.0, 3.0),
    color_clean="#1f4e79",
    color_noisy="#c0504d",
    linewidth_clean=1.0,
    linewidth_noisy=0.7,
    alpha_noisy=0.8,
    labels=None,
    ax=None,
):
    """
    Plot clean and noisy time series on a shared axis.

    The sample index is on the x-axis; both the original and the
    noisy signal are drawn on the y-axis for direct comparison.

    Parameters
    ----------
    clean : ndarray (1-D)
        Original signal.
    noisy : ndarray (1-D)
        Signal with added noise.
    sample : int or None
        Number of leading samples to display (None = all).
    xlabel, ylabel : str
        Axis labels.
    title : str or None
    figsize : tuple
    color_clean, color_noisy : str
        Line colors.
    linewidth_clean, linewidth_noisy : float
    alpha_noisy : float
        Transparency of the noisy line.
    labels : tuple of str or None
        (clean_label, noisy_label).
    ax : matplotlib Axes or None

    Returns
    -------
    ax : matplotlib Axes
    """
    if labels is None:
        labels = ("Clean", "Noisy")

    n = clean.shape[0] if sample is None else min(sample, clean.shape[0])
    t = np.arange(n)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(t, clean[:n], color=color_clean, linewidth=linewidth_clean,
            label=labels[0], zorder=3)
    ax.plot(t, noisy[:n], color=color_noisy, linewidth=linewidth_noisy,
            alpha=alpha_noisy, label=labels[1], zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best")

    return ax
