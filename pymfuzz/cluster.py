"""Fuzzy c-means soft clustering -- the core of Mfuzz.

Mfuzz's :func:`mfuzz` is a thin wrapper around ``e1071::cmeans`` with
``method = "cmeans"`` and Euclidean distance.  This module ports the
``cmeans`` algorithm faithfully, including

* the random-centre initialisation (``x[sample(1:xrows, ncenters), ]``),
* the row-permutation (``perm <- sample(xrows)``) of e1071's R wrapper,
* the C-level update loop (``src/cmeans.c``):

  - dissimilarities  ``d_ij = ||x_i - p_j||^2`` (squared Euclidean),
  - memberships      ``u_ij = (1/d_ij^(1/(m-1))) / sum_k(...)``,
  - prototypes       ``p_j = sum_i(u_ij^m x_i) / sum_i(u_ij^m)``,
  - objective        ``E = sum_ij u_ij^m d_ij`` (with weights),
  - convergence      ``|E_old - E_new| < reltol * (E_old + reltol)``.

The defaults match e1071 1.7.17 / Mfuzz 2.66.0: ``iter.max = 100``,
``reltol = sqrt(.Machine$double.eps)``, unit weights.

Because the algorithm is initialised at random, results are *not* bit-exact
across RNGs; instead pymfuzz reproduces e1071's exact arithmetic so that,
given identical initial centres and permutation, it converges to the same
solution.  Cross-implementation parity is assessed with ARI / membership
correlation (see ``tests/test_r_parity.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .dataset import as_expression_matrix

__all__ = ["FClust", "cmeans", "mfuzz"]

# R's sqrt(.Machine$double.eps)
_RELTOL = float(np.sqrt(np.finfo(np.float64).eps))


@dataclass
class FClust:
    """Result of fuzzy c-means clustering (mirrors e1071's ``fclust``).

    Attributes
    ----------
    centers : numpy.ndarray
        ``(c, timepoints)`` cluster prototype matrix.
    membership : numpy.ndarray
        ``(genes, c)`` soft membership matrix; rows sum to 1.
    cluster : numpy.ndarray
        ``(genes,)`` hard assignment -- the **1-based** index of the
        highest-membership cluster (matching R's ``which.max``).
    size : numpy.ndarray
        ``(c,)`` number of genes hard-assigned to each cluster.
    iter : int
        Number of iterations run (``retval$iter - 1`` in e1071).
    withinerror : float
        Final value of the fuzzy objective function.
    gene_names : list of str
    time_names : list of str
    cluster_names : list of str
    """

    centers: np.ndarray
    membership: np.ndarray
    cluster: np.ndarray
    size: np.ndarray
    iter: int
    withinerror: float
    gene_names: List[str] = field(default_factory=list)
    time_names: List[str] = field(default_factory=list)
    cluster_names: List[str] = field(default_factory=list)

    @property
    def n_clusters(self) -> int:
        return self.centers.shape[0]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"FClust(n_clusters={self.n_clusters}, "
            f"n_genes={self.membership.shape[0]}, iter={self.iter}, "
            f"withinerror={self.withinerror:.6g})"
        )


# ----------------------------------------------------------------------
# C-level kernels (vectorised numpy ports of src/cmeans.c)
# ----------------------------------------------------------------------
def _dissimilarities(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Squared Euclidean ``d[i, j] = ||x_i - p_j||^2`` (cmeans_dissimilarities)."""
    # (n, 1, c-cols) - (1, c, cols)  ->  (n, c)
    diff = x[:, None, :] - p[None, :, :]
    return np.sum(diff * diff, axis=2)


def _memberships(d: np.ndarray, exponent: float) -> np.ndarray:
    """Membership update -- faithful port of ``cmeans_memberships``.

    ``exponent = 1 / (m - 1)``.  Rows with one or more zero dissimilarities
    are handled exactly as the C code does (uniform 1/n_zeroes split).
    """
    n, c = d.shape
    u = np.empty((n, c), dtype=np.float64)
    zero_mask = d == 0.0
    n_zeroes = zero_mask.sum(axis=1)

    has_zero = n_zeroes > 0
    if np.any(has_zero):
        idx = np.where(has_zero)[0]
        for i in idx:
            v = 1.0 / n_zeroes[i]
            u[i, :] = np.where(zero_mask[i, :], v, 0.0)

    normal = ~has_zero
    if np.any(normal):
        dd = d[normal, :]
        v = 1.0 / np.power(dd, exponent)
        s = v.sum(axis=1, keepdims=True)
        u[normal, :] = v / s
    return u


def _prototypes(
    x: np.ndarray, u: np.ndarray, w: np.ndarray, f: float
) -> np.ndarray:
    """Prototype (centroid) update -- ``cmeans_prototypes`` (Euclidean)."""
    # v[i, j] = w_i * u_ij^f
    v = w[:, None] * np.power(u, f)          # (n, c)
    num = v.T @ x                             # (c, cols)
    den = v.sum(axis=0)[:, None]              # (c, 1)
    return num / den


def _error_fn(
    u: np.ndarray, d: np.ndarray, w: np.ndarray, f: float
) -> float:
    """Fuzzy objective ``sum_ij w_i u_ij^f d_ij`` -- ``cmeans_error_fn``."""
    return float(np.sum(w[:, None] * np.power(u, f) * d))


def _cmeans_core(
    x: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    m: float,
    itermax: int,
    reltol: float,
    verbose: bool,
):
    """Direct port of the C function ``cmeans`` in e1071/src/cmeans.c.

    ``x`` is already row-permuted; ``p`` are the initial centres; ``w`` are
    normalised weights.  Returns ``(centers, u, iter, ermin)``.
    """
    exponent = 1.0 / (m - 1.0)
    p = p.copy()

    d = _dissimilarities(x, p)
    u = _memberships(d, exponent)
    old_value = new_value = _error_fn(u, d, w, m)

    it = 0
    while True:
        it += 1
        if it > itermax:
            break
        p = _prototypes(x, u, w, m)
        d = _dissimilarities(x, p)
        u = _memberships(d, exponent)
        new_value = _error_fn(u, d, w, m)
        if abs(old_value - new_value) < reltol * (old_value + reltol):
            if verbose:
                print(f"Iteration: {it:3d} converged, "
                      f"Error: {new_value:.10f}")
            break
        if verbose:
            print(f"Iteration: {it:3d}, Error: {new_value:.10f}")
        old_value = new_value

    ermin = new_value
    return p, u, it, ermin


def cmeans(
    x,
    centers,
    m: float = 2.0,
    iter_max: int = 100,
    reltol: Optional[float] = None,
    weights=1.0,
    verbose: bool = False,
    random_state: Optional[int] = None,
) -> FClust:
    """Fuzzy c-means clustering -- a faithful port of ``e1071::cmeans``.

    Parameters
    ----------
    x : array-like | DataFrame | AnnData | ExpressionMatrix
        ``genes x timepoints`` data matrix; clustering is over rows.
    centers : int or numpy.ndarray
        Number of clusters, or an explicit ``(c, timepoints)`` initial
        centre matrix.  When an integer, centres are drawn at random from
        distinct data rows (as e1071 does).
    m : float, default 2.0
        Fuzzifier (``m > 1``).
    iter_max : int, default 100
        Maximum number of iterations.
    reltol : float, optional
        Relative convergence tolerance; defaults to ``sqrt(eps)`` as in R.
    weights : float or array-like, default 1.0
        Per-object weights (recycled, then normalised to sum to 1).
    verbose : bool, default False
        Print per-iteration error.
    random_state : int, optional
        Seed for the centre selection / row-permutation RNG.

    Returns
    -------
    FClust
    """
    em = as_expression_matrix(x)
    data = em.values
    xrows, xcols = data.shape
    rng = np.random.default_rng(random_state)

    if reltol is None:
        reltol = _RELTOL
    if reltol <= 0:
        raise ValueError("reltol must be positive.")
    if iter_max < 1:
        raise ValueError("iter_max must be positive.")

    # --- initial centres ---------------------------------------------
    if np.isscalar(centers):
        ncenters = int(centers)
        sel = rng.choice(xrows, size=ncenters, replace=False)
        p0 = data[sel, :].copy()
        # e1071 re-draws from unique rows when duplicates appear.
        uniq = np.unique(p0, axis=0)
        if uniq.shape[0] < ncenters:
            cn = np.unique(data, axis=0)
            if cn.shape[0] < ncenters:
                raise ValueError(
                    "More cluster centers than distinct data points."
                )
            sel2 = rng.choice(cn.shape[0], size=ncenters, replace=False)
            p0 = cn[sel2, :].copy()
    else:
        p0 = np.asarray(centers, dtype=np.float64)
        if p0.ndim != 2:
            raise ValueError("centers matrix must be 2-D.")
        ncenters = p0.shape[0]
        if np.unique(p0, axis=0).shape[0] != ncenters:
            raise ValueError("Initial centers are not distinct.")
        if xrows < ncenters:
            raise ValueError("More cluster centers than data points.")
    if p0.shape[1] != xcols:
        raise ValueError(
            "Must have same number of columns in 'x' and 'centers'."
        )

    # --- weights ------------------------------------------------------
    w = np.asarray(weights, dtype=np.float64).ravel()
    if np.any(w < 0):
        raise ValueError("weights has negative elements.")
    if not np.any(w > 0):
        raise ValueError("weights has no positive elements.")
    w = np.resize(w, xrows).astype(np.float64)
    w = w / w.sum()

    # --- row permutation (e1071's 'perm <- sample(xrows)') -----------
    perm = rng.permutation(xrows)
    x_perm = data[perm, :]
    w_perm = w[perm]

    centers_out, u_perm, it, ermin = _cmeans_core(
        x_perm, p0, w_perm, m, iter_max, reltol, verbose
    )

    # undo the permutation (R: 'u <- u[order(perm), ]')
    inv = np.argsort(perm)
    u = u_perm[inv, :]

    iter_reported = it - 1  # e1071: 'iter <- retval$iter - 1'
    cluster = np.argmax(u, axis=1) + 1  # 1-based, like which.max
    size = np.array(
        [int(np.sum(cluster == j + 1)) for j in range(ncenters)]
    )

    return FClust(
        centers=centers_out,
        membership=u,
        cluster=cluster,
        size=size,
        iter=iter_reported,
        withinerror=ermin,
        gene_names=list(em.gene_names),
        time_names=list(em.time_names),
        cluster_names=[str(j + 1) for j in range(ncenters)],
    )


def mfuzz(
    x,
    c: int,
    m: float,
    iter_max: int = 100,
    random_state: Optional[int] = None,
    **kwargs,
) -> FClust:
    """Soft clustering of time-series expression -- Mfuzz ``mfuzz``.

    Direct equivalent of R's ``mfuzz(eset, centers, m, ...)``, which calls
    ``cmeans(exprs(eset), centers = centers, method = "cmeans", m = m)``.

    Parameters
    ----------
    x : array-like | DataFrame | AnnData | ExpressionMatrix
        Standardised ``genes x timepoints`` expression (see
        :func:`pymfuzz.standardise`).
    c : int
        Number of clusters.
    m : float
        Fuzzifier (``m > 1``); estimate with :func:`pymfuzz.mestimate`.
    iter_max : int, default 100
        Maximum iterations.
    random_state : int, optional
        RNG seed for reproducible initialisation.
    **kwargs
        Forwarded to :func:`cmeans` (e.g. ``weights``, ``reltol``,
        ``verbose``).

    Returns
    -------
    FClust
    """
    return cmeans(
        x,
        centers=c,
        m=m,
        iter_max=iter_max,
        random_state=random_state,
        **kwargs,
    )
