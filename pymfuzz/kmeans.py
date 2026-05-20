"""Hard-clustering alternative -- Mfuzz ``kmeans2``.

R's ``kmeans2`` is a thin wrapper around ``stats::kmeans``; this module
provides a comparable Lloyd/Hartigan-style k-means so users have the
hard-clustering counterpart to :func:`pymfuzz.mfuzz`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .dataset import as_expression_matrix

__all__ = ["KMeansResult", "kmeans2"]


@dataclass
class KMeansResult:
    """Result of :func:`kmeans2` (mirrors R's ``kmeans`` object).

    Attributes
    ----------
    cluster : numpy.ndarray
        ``(genes,)`` **1-based** hard assignment.
    centers : numpy.ndarray
        ``(k, timepoints)`` cluster means.
    size : numpy.ndarray
        ``(k,)`` cluster sizes.
    withinss : numpy.ndarray
        ``(k,)`` within-cluster sum of squares.
    tot_withinss : float
        Total within-cluster sum of squares.
    iter : int
        Number of Lloyd iterations run.
    gene_names : list of str
    time_names : list of str
    """

    cluster: np.ndarray
    centers: np.ndarray
    size: np.ndarray
    withinss: np.ndarray
    tot_withinss: float
    iter: int
    gene_names: List[str] = field(default_factory=list)
    time_names: List[str] = field(default_factory=list)

    @property
    def n_clusters(self) -> int:
        return self.centers.shape[0]


def kmeans2(
    x,
    k: int,
    iter_max: int = 100,
    n_init: int = 1,
    random_state: Optional[int] = None,
) -> KMeansResult:
    """Hard k-means clustering of a time-course -- Mfuzz ``kmeans2``.

    Parameters
    ----------
    x : array-like | DataFrame | AnnData | ExpressionMatrix
        ``genes x timepoints`` expression input.
    k : int
        Number of clusters.
    iter_max : int, default 100
        Maximum Lloyd iterations (R's ``iter.max``).
    n_init : int, default 1
        Number of random restarts; the lowest-WSS solution is kept.
    random_state : int, optional
        RNG seed.

    Returns
    -------
    KMeansResult
    """
    em = as_expression_matrix(x)
    data = em.values
    n = data.shape[0]
    rng = np.random.default_rng(random_state)

    best: Optional[KMeansResult] = None
    for _ in range(max(1, n_init)):
        sel = rng.choice(n, size=k, replace=False)
        centers = data[sel, :].copy()
        labels = np.zeros(n, dtype=int)
        it = 0
        for it in range(1, iter_max + 1):
            dist = np.sum(
                (data[:, None, :] - centers[None, :, :]) ** 2, axis=2
            )
            new_labels = np.argmin(dist, axis=1)
            if np.array_equal(new_labels, labels) and it > 1:
                labels = new_labels
                break
            labels = new_labels
            for j in range(k):
                m = labels == j
                if m.any():
                    centers[j, :] = data[m, :].mean(axis=0)
        withinss = np.array(
            [
                float(
                    np.sum(
                        (data[labels == j] - centers[j]) ** 2
                    )
                )
                for j in range(k)
            ]
        )
        tot = float(withinss.sum())
        if best is None or tot < best.tot_withinss:
            best = KMeansResult(
                cluster=labels + 1,
                centers=centers.copy(),
                size=np.array(
                    [int(np.sum(labels == j)) for j in range(k)]
                ),
                withinss=withinss,
                tot_withinss=tot,
                iter=it,
                gene_names=list(em.gene_names),
                time_names=list(em.time_names),
            )
    assert best is not None
    return best
