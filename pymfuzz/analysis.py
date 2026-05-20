"""Diagnostic and post-processing routines -- Mfuzz analysis functions.

Faithful ports of ``mestimate``, ``acore``, ``Dmin``, ``cselection``,
``partcoef`` and ``overlap`` from the Mfuzz R package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .cluster import FClust, mfuzz
from .dataset import as_expression_matrix

__all__ = [
    "mestimate",
    "acore",
    "AcoreCluster",
    "Dmin",
    "cselection",
    "partcoef",
    "PartcoefResult",
    "overlap",
]


# ----------------------------------------------------------------------
# mestimate -- Schwammle & Jensen (2010) fuzzifier estimate
# ----------------------------------------------------------------------
def mestimate(x) -> float:
    """Estimate the fuzzifier ``m`` -- Mfuzz ``mestimate``.

    Implements the Schwammle & Jensen (2010) heuristic exactly::

        N    = nrow(exprs(eset))      # number of genes
        D    = ncol(exprs(eset))      # number of timepoints
        m.sj = 1 + (1418/N + 22.05) * D^(-2)
                 + (12.33/N + 0.243) * D^(-0.0406*log(N) - 0.1134)

    Parameters
    ----------
    x : array-like | DataFrame | AnnData | ExpressionMatrix
        ``genes x timepoints`` expression input.

    Returns
    -------
    float
        The estimated fuzzifier ``m``.
    """
    em = as_expression_matrix(x)
    N, D = em.values.shape
    m_sj = (
        1.0
        + (1418.0 / N + 22.05) * D ** (-2.0)
        + (12.33 / N + 0.243)
        * D ** (-0.0406 * np.log(N) - 0.1134)
    )
    return float(m_sj)


# ----------------------------------------------------------------------
# acore -- core genes per cluster
# ----------------------------------------------------------------------
@dataclass
class AcoreCluster:
    """Core genes of a single cluster (one element of :func:`acore`).

    Attributes
    ----------
    cluster : int
        1-based cluster index.
    names : list of str
        Gene names with membership above the threshold, sorted by
        descending membership.
    membership : numpy.ndarray
        Matching membership values.
    """

    cluster: int
    names: List[str]
    membership: np.ndarray

    def to_dataframe(self):
        """Return a ``NAME / MEM.SHIP`` DataFrame (as R's ``acore`` does)."""
        import pandas as pd

        return pd.DataFrame(
            {"NAME": self.names, "MEM.SHIP": self.membership}
        )

    def __len__(self) -> int:
        return len(self.names)


def acore(
    x, cl: FClust, min_acore: float = 0.5
) -> List[AcoreCluster]:
    """Extract per-cluster "core" genes -- Mfuzz ``acore``.

    For each cluster ``j`` a gene is in the core when it is hard-assigned
    to ``j`` *and* its membership in ``j`` exceeds ``min_acore``.  Unlike
    the R function, the returned cores are **sorted by descending
    membership** for convenience.

    Parameters
    ----------
    x : array-like | DataFrame | AnnData | ExpressionMatrix
        The expression matrix that was clustered (used for gene names).
    cl : FClust
        A clustering result from :func:`pymfuzz.mfuzz`.
    min_acore : float, default 0.5
        Membership threshold.

    Returns
    -------
    list of AcoreCluster
        One entry per cluster, in cluster order.
    """
    em = as_expression_matrix(x)
    gene_names = (
        cl.gene_names if cl.gene_names else list(em.gene_names)
    )
    gene_names = np.asarray(gene_names, dtype=object)
    out: List[AcoreCluster] = []
    for j in range(cl.n_clusters):
        mask = (cl.cluster == j + 1) & (cl.membership[:, j] > min_acore)
        mem = cl.membership[mask, j]
        names = gene_names[mask]
        order = np.argsort(-mem, kind="stable")
        out.append(
            AcoreCluster(
                cluster=j + 1,
                names=[str(n) for n in names[order]],
                membership=mem[order].copy(),
            )
        )
    return out


# ----------------------------------------------------------------------
# Dmin -- minimum centroid distance vs cluster number
# ----------------------------------------------------------------------
def _pairwise_min_dist(centers: np.ndarray) -> float:
    """``min(dist(centers))`` -- minimum pairwise Euclidean distance."""
    c = centers.shape[0]
    best = np.inf
    for i in range(c):
        for j in range(i + 1, c):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if d < best:
                best = d
    return float(best)


def Dmin(
    x,
    m: float,
    crange: Sequence[int] = range(4, 41, 4),
    repeats: int = 3,
    visu: bool = False,
    random_state: Optional[int] = None,
):
    """Mean minimum-centroid distance vs cluster count -- Mfuzz ``Dmin``.

    The classic elbow diagnostic for choosing the number of clusters: for
    every candidate ``c`` in ``crange`` the data is clustered ``repeats``
    times and ``min(dist(centers))`` recorded; the average over repeats is
    returned.

    Parameters
    ----------
    x : array-like | DataFrame | AnnData | ExpressionMatrix
    m : float
        Fuzzifier.
    crange : sequence of int, default range(4, 41, 4)
        Candidate cluster counts.
    repeats : int, default 3
        Number of runs per cluster count.
    visu : bool, default False
        If True, return ``(curve, fig)`` with the diagnostic plot.
    random_state : int, optional
        Base RNG seed (each run uses a derived seed).

    Returns
    -------
    numpy.ndarray or (numpy.ndarray, matplotlib.figure.Figure)
        The mean minimum-centroid distance for each ``c`` in ``crange``.
    """
    em = as_expression_matrix(x)
    crange = list(crange)
    DminM = np.zeros((len(crange), repeats))
    seed = 0
    for ii in range(repeats):
        for j, c in enumerate(crange):
            rs = (
                None
                if random_state is None
                else random_state + seed
            )
            seed += 1
            cl = mfuzz(em, c=c, m=m, random_state=rs)
            DminM[j, ii] = _pairwise_min_dist(cl.centers)
    curve = DminM.mean(axis=1)
    if visu:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(crange, curve, "o-", color="black")
        ax.set_xlabel("Cluster number")
        ax.set_ylabel("Min. centroid distance")
        return curve, fig
    return curve


# ----------------------------------------------------------------------
# cselection -- non-empty cluster frequency vs cluster number
# ----------------------------------------------------------------------
def cselection(
    x,
    m: float,
    crange: Sequence[int] = range(4, 33, 4),
    repeats: int = 5,
    visu: bool = False,
    random_state: Optional[int] = None,
):
    """Empty-cluster diagnostic -- Mfuzz ``cselection``.

    For each candidate ``c`` and each of ``repeats`` runs, counts the
    number of clusters that contain at least one gene with membership
    ``> 0.5``.  Cluster counts at which empty clusters appear repeatedly
    are likely too large.

    Parameters
    ----------
    x : array-like | DataFrame | AnnData | ExpressionMatrix
    m : float
        Fuzzifier.
    crange : sequence of int, default range(4, 33, 4)
    repeats : int, default 5
    visu : bool, default False
        If True, also return the diagnostic figure.
    random_state : int, optional

    Returns
    -------
    numpy.ndarray (repeats x len(crange))
        Number of non-empty clusters; optionally with the figure.
    """
    em = as_expression_matrix(x)
    crange = list(crange)
    Nonempty = np.zeros((repeats, len(crange)))
    seed = 0
    for i, c in enumerate(crange):
        for ii in range(repeats):
            rs = (
                None if random_state is None else random_state + seed
            )
            seed += 1
            U = mfuzz(em, c=c, m=m, random_state=rs).membership
            Nonempty[ii, i] = int(np.sum((U > 0.5).sum(axis=0) > 0))
    if visu:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for i in range(repeats):
            ax.plot(crange, Nonempty[i, :], "x", color="black")
        lim = max(crange) + 1
        ax.plot([0, lim], [0, lim], color="red")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Number of non-empty clusters")
        return Nonempty, fig
    return Nonempty


# ----------------------------------------------------------------------
# partcoef -- partition coefficient
# ----------------------------------------------------------------------
@dataclass
class PartcoefResult:
    """Partition-coefficient grid -- result of :func:`partcoef`.

    Attributes
    ----------
    F : numpy.ndarray
        ``(len(crange), len(mrange))`` partition coefficient
        ``sum(U^2) / (n_genes * c)``.
    F_n : numpy.ndarray
        ``F`` minus its theoretical minimum (``F - F_min``).
    F_min : numpy.ndarray
        Theoretical minimum ``1 / c^2``.
    crange : list of int
    mrange : list of float
    """

    F: np.ndarray
    F_n: np.ndarray
    F_min: np.ndarray
    crange: List[int]
    mrange: List[float]


def partcoef(
    x,
    crange: Sequence[int] = range(4, 33, 4),
    mrange: Sequence[float] = np.arange(1.05, 2.001, 0.1),
    random_state: Optional[int] = None,
) -> PartcoefResult:
    """Partition coefficient over a ``(c, m)`` grid -- Mfuzz ``partcoef``.

    The partition coefficient ``F = sum(U^2) / (n*c)`` measures the
    crispness of a fuzzy partition; ``F_min = 1/c^2`` is the value for a
    maximally fuzzy (uniform) partition.

    Parameters
    ----------
    x : array-like | DataFrame | AnnData | ExpressionMatrix
    crange : sequence of int, default range(4, 33, 4)
    mrange : sequence of float, default arange(1.05, 2.0, 0.1)
    random_state : int, optional

    Returns
    -------
    PartcoefResult
    """
    em = as_expression_matrix(x)
    crange = list(crange)
    mrange = [float(v) for v in mrange]
    F = np.full((len(crange), len(mrange)), np.nan)
    F_n = np.full_like(F, np.nan)
    F_min = np.full_like(F, np.nan)
    seed = 0
    for i, c in enumerate(crange):
        for j, m in enumerate(mrange):
            rs = (
                None if random_state is None else random_state + seed
            )
            seed += 1
            U = mfuzz(em, c=c, m=m, random_state=rs).membership
            F[i, j] = np.sum(U ** 2) / (U.shape[0] * U.shape[1])
            F_min[i, j] = 1.0 / c ** 2
            F_n[i, j] = F[i, j] - F_min[i, j]
    return PartcoefResult(F, F_n, F_min, crange, mrange)


# ----------------------------------------------------------------------
# overlap -- cluster-to-cluster membership overlap
# ----------------------------------------------------------------------
def overlap(cl: FClust) -> np.ndarray:
    """Cluster-to-cluster membership overlap -- Mfuzz ``overlap``.

    Computes ``O[i, j] = sum_g(u_gi * u_gj)`` then column-normalises so
    that every column sums to 1.

    Parameters
    ----------
    cl : FClust
        A clustering result.

    Returns
    -------
    numpy.ndarray
        The ``(c, c)`` column-normalised overlap matrix.
    """
    U = cl.membership
    O = U.T @ U  # O[i, j] = sum_g u_gi u_gj
    O = O / O.sum(axis=0, keepdims=True)
    return O
