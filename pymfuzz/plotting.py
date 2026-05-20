"""Matplotlib plots -- ports of Mfuzz's signature figures.

* :func:`mfuzz_plot` / :func:`mfuzz_plot2` -- the soft-cluster
  temporal-profile grid: per-cluster line plots where each gene line is
  coloured by its membership value (low membership -> grey/orange,
  high membership -> red/magenta).
* :func:`kmeans2_plot` -- the hard-clustering counterpart.
* :func:`overlap_plot` -- a PCA projection of cluster centres with
  overlap-weighted connecting edges.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .cluster import FClust
from .dataset import as_expression_matrix
from .kmeans import KMeansResult

__all__ = [
    "mfuzz_plot",
    "mfuzz_plot2",
    "kmeans2_plot",
    "overlap_plot",
    "mfuzz_colorbar",
]


# Mfuzz's default colour ramps (taken verbatim from the R source).
_MFUZZ_COLO = [
    "#FF8F00", "#FFA700", "#FFBF00", "#FFD700", "#FFEF00", "#F7FF00",
    "#DFFF00", "#C7FF00", "#AFFF00", "#97FF00", "#80FF00", "#68FF00",
    "#50FF00", "#38FF00", "#20FF00", "#08FF00", "#00FF10", "#00FF28",
    "#00FF40", "#00FF58", "#00FF70", "#00FF87", "#00FF9F", "#00FFB7",
    "#00FFCF", "#00FFE7", "#00FFFF", "#00E7FF", "#00CFFF", "#00B7FF",
    "#009FFF", "#0087FF", "#0070FF", "#0058FF", "#0040FF", "#0028FF",
    "#0010FF", "#0800FF", "#2000FF", "#3800FF", "#5000FF", "#6800FF",
    "#8000FF", "#9700FF", "#AF00FF", "#C700FF", "#DF00FF", "#F700FF",
    "#FF00EF", "#FF00D7", "#FF00BF", "#FF00A7", "#FF008F", "#FF0078",
    "#FF0060", "#FF0048", "#FF0030", "#FF0018",
]

# mfuzz.plot2's ramp -- starts at pure red.
_MFUZZ_COLO2 = [
    "#FF0000", "#FF1800", "#FF3000", "#FF4800", "#FF6000", "#FF7800",
] + _MFUZZ_COLO


def _grid(n: int, mfrow: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if mfrow is not None:
        return mfrow
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    return nrow, ncol


def _draw_clusters(
    eset,
    cl: FClust,
    colo: Sequence[str],
    mfrow: Optional[Tuple[int, int]],
    min_mem: float,
    time_labels: Optional[Sequence],
    time_points: Optional[Sequence],
    ylim_set: Optional[Tuple[float, float]],
    xlab: str,
    ylab: str,
    centre: bool,
    centre_col: str,
    centre_lwd: float,
    figsize: Optional[Tuple[float, float]],
):
    """Shared engine for :func:`mfuzz_plot` / :func:`mfuzz_plot2`."""
    import matplotlib.pyplot as plt

    em = as_expression_matrix(eset)
    data = em.values
    n_time = data.shape[1]
    cidx = cl.cluster
    memship = cl.membership.copy()
    memship[memship < min_mem] = -1.0
    n_clusters = cl.n_clusters

    colorseq = np.linspace(0.0, 1.0, len(colo))
    nrow, ncol = _grid(n_clusters, mfrow)
    if figsize is None:
        figsize = (3.0 * ncol, 2.6 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    xs = (
        np.asarray(time_points, dtype=float)
        if time_points is not None
        else np.arange(1, n_time + 1)
    )

    for j in range(n_clusters):
        ax = axes_flat[j]
        sel = cidx == (j + 1)
        tmp = data[sel, :]
        tmpmem = memship[sel, j]
        if tmp.shape[0] == 0:
            ymin, ymax = -1.0, 1.0
        else:
            ymin, ymax = float(np.min(tmp)), float(np.max(tmp))
        if ylim_set is not None:
            ymin, ymax = ylim_set
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"Cluster {j + 1}")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        if time_labels is not None:
            labels = [str(t) for t in time_labels]
            n = len(xs)
            # thin the ticks when crowded so labels do not overlap, and
            # rotate them — narrow faceted panels cannot fit many labels
            step = max(1, int(np.ceil(n / 8)))
            show = list(range(0, n, step))
            if show and show[-1] != n - 1:
                show.append(n - 1)
            ax.set_xticks([xs[i] for i in show])
            ax.set_xticklabels([labels[i] for i in show],
                               rotation=45, ha="right", fontsize=8)
        else:
            ax.tick_params(axis="x", labelsize=8)

        # draw lines bucketed by membership colour band (as Mfuzz does)
        if tmp.shape[0] > 0:
            for jj in range(len(colorseq) - 1):
                band = (tmpmem >= colorseq[jj]) & (
                    tmpmem <= colorseq[jj + 1]
                )
                if band.any():
                    for row in tmp[band, :]:
                        ax.plot(xs, row, color=colo[jj], linewidth=0.8)
        if centre:
            ax.plot(
                xs,
                cl.centers[j, :],
                color=centre_col,
                linewidth=centre_lwd,
            )

    for k in range(n_clusters, len(axes_flat)):
        axes_flat[k].axis("off")
    fig.tight_layout()
    return fig


def mfuzz_plot(
    eset,
    cl: FClust,
    mfrow: Optional[Tuple[int, int]] = None,
    colo: Optional[Sequence[str]] = None,
    min_mem: float = 0.0,
    time_labels: Optional[Sequence] = None,
    new_window: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
):
    """Soft-cluster temporal-profile grid -- Mfuzz ``mfuzz.plot``.

    Produces the signature Mfuzz figure: one panel per cluster, each gene
    drawn as a line coloured by its membership in that cluster (low ->
    orange, high -> magenta).

    Parameters
    ----------
    eset : array-like | DataFrame | AnnData | ExpressionMatrix
        The (standardised) expression matrix that was clustered.
    cl : FClust
        Clustering result from :func:`pymfuzz.mfuzz`.
    mfrow : (int, int), optional
        Panel grid ``(rows, cols)``; auto-sized when omitted.
    colo : sequence of str, optional
        Colour ramp; defaults to Mfuzz's orange->magenta palette.
    min_mem : float, default 0.0
        Genes with membership below this are not drawn.
    time_labels : sequence, optional
        Custom x-axis tick labels.
    new_window : bool, default False
        Kept for R-signature compatibility (ignored).
    figsize : (float, float), optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    return _draw_clusters(
        eset,
        cl,
        colo if colo is not None else _MFUZZ_COLO,
        mfrow,
        min_mem,
        time_labels,
        None,
        None,
        "Time",
        "Expression changes",
        False,
        "black",
        2.0,
        figsize,
    )


def mfuzz_plot2(
    eset,
    cl: FClust,
    mfrow: Optional[Tuple[int, int]] = None,
    colo: Optional[Sequence[str]] = None,
    min_mem: float = 0.0,
    time_labels: Optional[Sequence] = None,
    time_points: Optional[Sequence] = None,
    ylim_set: Optional[Tuple[float, float]] = None,
    xlab: str = "Time",
    ylab: str = "Expression changes",
    centre: bool = False,
    centre_col: str = "black",
    centre_lwd: float = 2.0,
    figsize: Optional[Tuple[float, float]] = None,
):
    """Enhanced soft-cluster profile grid -- Mfuzz ``mfuzz.plot2``.

    Like :func:`mfuzz_plot` but exposes explicit ``time_points``, a shared
    ``ylim_set`` and an optional cluster-centroid overlay.

    Parameters
    ----------
    eset : array-like | DataFrame | AnnData | ExpressionMatrix
    cl : FClust
    mfrow : (int, int), optional
    colo : sequence of str, optional
        Colour ramp; defaults to mfuzz.plot2's red->magenta palette.
        Pass ``"fancy"`` for the wide RGB sweep.
    min_mem : float, default 0.0
    time_labels : sequence, optional
    time_points : sequence, optional
        Numeric x positions for the timepoints.
    ylim_set : (float, float), optional
        Shared y-axis limits across all panels.
    xlab, ylab : str
    centre : bool, default False
        Overlay each cluster centroid.
    centre_col : str, default "black"
    centre_lwd : float, default 2.0
    figsize : (float, float), optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(colo, str) and colo == "fancy":
        b = list(range(255, -1, -1)) + [0] * 256 + [0] * 106
        g = list(range(256)) + list(range(255, -1, -1)) + [0] * 106
        r = list(range(256)) + [255] * 256 + list(range(255, 149, -1))
        colo = [
            (rr / 255.0, gg / 255.0, bb / 255.0)
            for rr, gg, bb in zip(r, g, b)
        ]
    return _draw_clusters(
        eset,
        cl,
        colo if colo is not None else _MFUZZ_COLO2,
        mfrow,
        min_mem,
        time_labels,
        time_points,
        ylim_set,
        xlab,
        ylab,
        centre,
        centre_col,
        centre_lwd,
        figsize,
    )


def kmeans2_plot(
    eset,
    kl: KMeansResult,
    mfrow: Optional[Tuple[int, int]] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """Hard-cluster temporal-profile grid -- Mfuzz ``kmeans2.plot``.

    Parameters
    ----------
    eset : array-like | DataFrame | AnnData | ExpressionMatrix
    kl : KMeansResult
        Result of :func:`pymfuzz.kmeans2`.
    mfrow : (int, int), optional
    figsize : (float, float), optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    em = as_expression_matrix(eset)
    data = em.values
    n_time = data.shape[1]
    cidx = kl.cluster
    k = kl.n_clusters
    nrow, ncol = _grid(k, mfrow)
    if figsize is None:
        figsize = (3.0 * ncol, 2.6 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()
    xs = np.arange(1, n_time + 1)
    for j in range(k):
        ax = axes_flat[j]
        tmp = data[cidx == (j + 1), :]
        if tmp.shape[0] == 0:
            ymin, ymax = -1.0, 1.0
        else:
            ymin, ymax = float(np.min(tmp)), float(np.max(tmp))
        ax.set_xlim(1, n_time)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"Cluster {j + 1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Expression")
        for row in tmp:
            ax.plot(xs, row, color="black", linewidth=0.8)
    for kk in range(k, len(axes_flat)):
        axes_flat[kk].axis("off")
    fig.tight_layout()
    return fig


def overlap_plot(
    cl: FClust,
    ov: np.ndarray,
    thres: float = 0.1,
    magni: float = 30.0,
    figsize: Tuple[float, float] = (6.0, 6.0),
):
    """PCA overlap map of cluster centres -- Mfuzz ``overlap.plot``.

    Projects the cluster centroids onto their first two principal
    components and draws an edge between every pair of clusters whose
    overlap exceeds ``thres``, with line width proportional to overlap.

    Parameters
    ----------
    cl : FClust
        Clustering result.
    ov : numpy.ndarray
        Overlap matrix from :func:`pymfuzz.overlap`.
    thres : float, default 0.1
        Minimum overlap to draw an edge.
    magni : float, default 30.0
        Edge-width magnification factor.
    figsize : (float, float), default (6, 6)

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    centers = cl.centers
    # prcomp(centers, scale = TRUE): centre + scale, then SVD.
    mu = centers.mean(axis=0)
    sd = centers.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    z = (centers - mu) / sd
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    rot = vt.T  # loadings, columns are PCs
    # R: x[[5]] <- t(t(rotation) %*% t(centers))  ==  centers %*% rotation
    proj = centers @ rot

    fig, ax = plt.subplots(figsize=figsize)
    nclust = proj.shape[0]
    for i in range(nclust):
        for j in range(nclust):
            if thres < ov[i, j]:
                ax.plot(
                    [proj[i, 0], proj[j, 0]],
                    [proj[i, 1], proj[j, 1]],
                    color="blue",
                    linewidth=magni * ov[i, j],
                )
    for i in range(nclust):
        ax.plot(proj[i, 0], proj[i, 1], "o", color="red", markersize=18)
        ax.annotate(
            str(i + 1),
            (proj[i, 0], proj[i, 1]),
            ha="center",
            va="center",
            fontweight="bold",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    return fig


def mfuzz_colorbar(
    col: Optional[Sequence[str]] = None,
    horizontal: bool = False,
    ax=None,
    figsize: Tuple[float, float] = (1.4, 4.0),
    **kwargs,
):
    """Membership colour-scale bar -- Mfuzz ``mfuzzColorBar``.

    Draws the 0..1 membership colour key that accompanies the
    :func:`mfuzz_plot` / :func:`mfuzz_plot2` figures.  Faithful port of
    R's ``mfuzzColorBar`` (which delegates to ``marray::maColorBar``):
    the bar shows ``seq(0, 1, 0.01)`` filled with the colour ramp and is
    annotated with ``k = 11`` evenly spaced ticks.

    Parameters
    ----------
    col : sequence of str, optional
        Colour ramp.  Defaults to Mfuzz's orange->magenta palette; pass
        the string ``"fancy"`` for the wide RGB sweep used by
        :func:`mfuzz_plot2`.
    horizontal : bool, default False
        If True draw the bar horizontally; vertical otherwise (the R
        default).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure/axes is created when omitted.
    figsize : (float, float), default (1.4, 4.0)
        Figure size when ``ax`` is not supplied.
    **kwargs
        Forwarded to :meth:`~matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the colour bar.

    Examples
    --------
    >>> import pymfuzz as mf
    >>> fig = mf.mfuzz_colorbar()
    >>> fig is not None
    True
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if col is None:
        colors = list(_MFUZZ_COLO)
    elif isinstance(col, str) and col == "fancy":
        b = list(range(255, -1, -1)) + [0] * 256 + [0] * 106
        g = list(range(256)) + list(range(255, -1, -1)) + [0] * 106
        r = list(range(256)) + [255] * 256 + list(range(255, 149, -1))
        colors = [
            (rr / 255.0, gg / 255.0, bb / 255.0)
            for rr, gg, bb in zip(r, g, b)
        ]
    else:
        colors = list(col)

    cmap = LinearSegmentedColormap.from_list("mfuzz_bar", colors)
    # R: maColorBar(seq(0, 1, 0.01), ..., k = 11)
    scale = np.arange(0.0, 1.0001, 0.01)
    ticks = np.linspace(0.0, 1.0, 11)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    grad = scale.reshape(1, -1) if horizontal else scale.reshape(-1, 1)
    if horizontal:
        ax.imshow(
            grad, aspect="auto", cmap=cmap, origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0), **kwargs,
        )
        ax.set_yticks([])
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    else:
        ax.imshow(
            grad, aspect="auto", cmap=cmap, origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0), **kwargs,
        )
        ax.set_xticks([])
        ax.yaxis.tick_right()
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
    fig.tight_layout()
    return fig
