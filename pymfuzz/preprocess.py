"""Preprocessing routines -- faithful ports of Mfuzz's R/preprocess functions.

These mirror, line for line, the corresponding R functions in the Mfuzz
package (``standardise``, ``standardise2``, ``filter.NA``, ``fill.NA``,
``filter.std``, ``randomise``).  All use R's *sample* standard deviation
(``ddof=1``, denominator ``n - 1``) so that :func:`standardise` and
:func:`standardise2` are bit-exact against R.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .dataset import ExpressionMatrix, as_expression_matrix

__all__ = [
    "standardise",
    "standardise2",
    "filter_NA",
    "fill_NA",
    "filter_std",
    "randomise",
]


def _row_sd(row: np.ndarray) -> float:
    """R's ``sd(x, na.rm = TRUE)`` for a single vector (sample SD)."""
    vals = row[~np.isnan(row)]
    if vals.size < 2:
        return np.nan
    return float(np.std(vals, ddof=1))


def standardise(x) -> ExpressionMatrix:
    """Per-gene z-score standardisation -- Mfuzz ``standardise``.

    For every gene (row) ``i``::

        data[i, ] <- (data[i, ] - mean(data[i, ])) / sd(data[i, ])

    using NA-aware mean and *sample* SD.  This is the recommended Mfuzz
    preprocessing so that clustering captures expression *changes* rather
    than absolute amplitude.

    Parameters
    ----------
    x : array-like | DataFrame | AnnData | ExpressionMatrix
        ``genes x timepoints`` expression input.

    Returns
    -------
    ExpressionMatrix
        A new matrix with standardised rows.
    """
    em = as_expression_matrix(x).copy()
    data = em.values
    for i in range(data.shape[0]):
        row = data[i, :]
        mu = np.nanmean(row) if np.any(~np.isnan(row)) else np.nan
        sd = _row_sd(row)
        data[i, :] = (row - mu) / sd
    return em


def standardise2(x, timepoint: int = 1) -> ExpressionMatrix:
    """Standardise relative to one timepoint -- Mfuzz ``standardise2``.

    For every gene (row) ``i``::

        data[i, ] <- (data[i, ] - data[i, timepoint]) / sd(data[i, ])

    Parameters
    ----------
    x : array-like | DataFrame | AnnData | ExpressionMatrix
        ``genes x timepoints`` expression input.
    timepoint : int, default 1
        Reference timepoint.  **1-based**, matching the R argument.

    Returns
    -------
    ExpressionMatrix
    """
    em = as_expression_matrix(x).copy()
    data = em.values
    idx = timepoint - 1  # R is 1-based
    if not 0 <= idx < data.shape[1]:
        raise IndexError(
            f"timepoint {timepoint} out of range 1..{data.shape[1]}"
        )
    for i in range(data.shape[0]):
        row = data[i, :]
        sd = _row_sd(row)
        data[i, :] = (row - row[idx]) / sd
    return em


def filter_NA(x, thres: float = 0.25, verbose: bool = True) -> ExpressionMatrix:
    """Drop genes with too many missing values -- Mfuzz ``filter.NA``.

    A gene is *excluded* when its fraction of NA values **exceeds**
    ``thres``.

    Parameters
    ----------
    x : array-like | DataFrame | AnnData | ExpressionMatrix
    thres : float, default 0.25
        Maximum tolerated NA fraction.
    verbose : bool, default True
        Print the number of excluded genes (as the R function does).

    Returns
    -------
    ExpressionMatrix
        The filtered matrix.
    """
    em = as_expression_matrix(x)
    data = em.values
    ncol = data.shape[1]
    frac = np.isnan(data).sum(axis=1) / ncol
    excluded = frac > thres
    if verbose:
        print(f"{int(excluded.sum())} genes excluded.")
    keep = ~excluded
    return ExpressionMatrix(
        data[keep, :].copy(),
        [g for g, k in zip(em.gene_names, keep) if k],
        list(em.time_names),
    )


def fill_NA(x, mode: str = "mean", k: int = 10) -> ExpressionMatrix:
    """Impute missing values -- Mfuzz ``fill.NA``.

    Parameters
    ----------
    x : array-like | DataFrame | AnnData | ExpressionMatrix
    mode : {"mean", "median", "knn", "knnw"}, default "mean"
        * ``mean``  -- replace NAs with the gene's NA-aware mean.
        * ``median``-- replace NAs with the gene's NA-aware median.
        * ``knn``   -- average of the ``k`` nearest genes (Euclidean
          distance over non-missing columns).
        * ``knnw``  -- inverse-distance weighted average of ``k`` nearest
          genes.
    k : int, default 10
        Number of neighbours for ``knn``/``knnw``.

    Returns
    -------
    ExpressionMatrix
    """
    em = as_expression_matrix(x).copy()
    data = em.values
    n = data.shape[0]

    if mode == "mean":
        for i in range(n):
            row = data[i, :]
            m = np.isnan(row)
            if m.any():
                row[m] = np.nanmean(row)
        return em

    if mode == "median":
        for i in range(n):
            row = data[i, :]
            m = np.isnan(row)
            if m.any():
                row[m] = np.nanmedian(row)
        return em

    if mode in ("knn", "knnw"):
        datatmp = data.copy()
        for i in range(n):
            row = data[i, :]
            miss = np.isnan(row)
            if not miss.any():
                continue
            # squared Euclidean distance over the columns present in row i,
            # NA-aware (matches R's apply(..., na.rm = TRUE)).
            diff = row[None, :] - datatmp
            dist = np.sqrt(np.nansum(diff ** 2, axis=1))
            order = np.argsort(dist, kind="stable")
            nn = order[1:k + 1]  # exclude self (index 0)
            if mode == "knn":
                fill = np.nanmean(data[np.ix_(nn, np.where(miss)[0])], axis=0)
            else:  # knnw -- inverse-distance weights
                w = dist[nn].copy()
                with np.errstate(divide="ignore"):
                    W = 1.0 / w
                W = W / np.nansum(W)
                sub = data[np.ix_(nn, np.where(miss)[0])]
                fill = np.nansum(W[:, None] * sub, axis=0)
            row[miss] = fill
        return em

    raise ValueError(f"unknown fill mode {mode!r}")


def filter_std(
    x, min_std: float, visu: bool = False
) -> ExpressionMatrix:
    """Drop low-variability genes -- Mfuzz ``filter.std``.

    A gene is *kept* when its (sample) SD is **greater than** ``min_std``;
    genes with an NA SD are kept (matching R's ``index[is.na(index)] <-
    TRUE``).

    Parameters
    ----------
    x : array-like | DataFrame | AnnData | ExpressionMatrix
    min_std : float
        SD threshold.
    visu : bool, default False
        If True, return ``(matrix, fig)`` with the sorted-SD diagnostic
        plot instead of just ``matrix``.

    Returns
    -------
    ExpressionMatrix or (ExpressionMatrix, matplotlib.figure.Figure)
    """
    em = as_expression_matrix(x)
    data = em.values
    sds = np.array([_row_sd(data[i, :]) for i in range(data.shape[0])])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        index = sds > min_std
    index[np.isnan(sds)] = True
    print(f"{int((~index).sum())} genes excluded.")
    filtered = ExpressionMatrix(
        data[index, :].copy(),
        [g for g, kk in zip(em.gene_names, index) if kk],
        list(em.time_names),
    )
    if visu:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(np.sort(sds[~np.isnan(sds)]), ".", color="black")
        ax.set_xlabel("Ordered genes")
        ax.set_ylabel("Sd")
        return filtered, fig
    return filtered


def randomise(eset, random_state: Optional[int] = None) -> ExpressionMatrix:
    """Randomise a time-course by permuting within each gene -- Mfuzz ``randomise``.

    Direct port of R's ``randomise(eset)``.  For every gene (row) the
    expression values are independently shuffled across the time points::

        for each gene i:
            dataR[i, ] <- data[i, sample(ncol)]

    The per-gene value *set* is preserved exactly -- only the temporal
    ordering is destroyed.  Clustering the randomised data and comparing
    with the original (e.g. via :func:`pymfuzz.Dmin` or
    :func:`pymfuzz.overlap`) tells you whether the cluster structure seen
    in the real data exceeds what an arbitrary permutation produces.

    Parameters
    ----------
    x : array-like | DataFrame | AnnData | ExpressionMatrix
        ``genes x timepoints`` expression input.
    random_state : int, optional
        Seed for the per-gene permutation RNG (R relies on the global
        seed; pass an explicit seed here for reproducibility).

    Returns
    -------
    ExpressionMatrix
        A new matrix; each row is a permutation of the corresponding
        input row, with gene and timepoint labels preserved.

    Examples
    --------
    >>> import numpy as np, pymfuzz as mf
    >>> x = np.arange(12, dtype=float).reshape(3, 4)
    >>> r = mf.randomise(x, random_state=0)
    >>> all(set(r.values[i]) == set(x[i]) for i in range(3))
    True
    """
    em = as_expression_matrix(eset).copy()
    data = em.values
    rng = np.random.default_rng(random_state)
    ncol = data.shape[1]
    for i in range(data.shape[0]):
        data[i, :] = data[i, rng.permutation(ncol)]
    return em
