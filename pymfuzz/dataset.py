"""Input coercion for pymfuzz.

Mfuzz works on a Bioconductor :class:`ExpressionSet` (a *genes x timepoints*
expression matrix).  This module provides a single helper,
:func:`as_expression_matrix`, that coerces the various accepted Python
inputs into a uniform internal representation:

* a 2-D :class:`numpy.ndarray` of expression values (``float64``);
* a list of gene (row) names;
* a list of timepoint (column) names.

Accepted inputs
---------------
* :class:`numpy.ndarray` -- a plain ``genes x timepoints`` matrix.
* :class:`pandas.DataFrame` -- rows = genes, columns = timepoints.
* :class:`anndata.AnnData` -- ``X`` is ``genes x timepoints`` *as stored*
  (Mfuzz time-courses are usually small, so no transpose heuristics are
  applied -- the matrix is taken exactly as given).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

try:  # optional dependency, only needed for AnnData input
    import anndata as _ad
except Exception:  # pragma: no cover
    _ad = None

try:
    import pandas as _pd
except Exception:  # pragma: no cover
    _pd = None

__all__ = ["ExpressionMatrix", "as_expression_matrix"]


@dataclass
class ExpressionMatrix:
    """A simple ``genes x timepoints`` container (the Python ExpressionSet).

    Attributes
    ----------
    values : numpy.ndarray
        2-D ``float64`` array, rows = genes, columns = timepoints.
    gene_names : list of str
        Row labels.
    time_names : list of str
        Column labels.
    """

    values: np.ndarray
    gene_names: List[str]
    time_names: List[str]

    @property
    def n_genes(self) -> int:
        return self.values.shape[0]

    @property
    def n_time(self) -> int:
        return self.values.shape[1]

    @property
    def shape(self):
        return self.values.shape

    def copy(self) -> "ExpressionMatrix":
        return ExpressionMatrix(
            self.values.copy(), list(self.gene_names), list(self.time_names)
        )

    def to_dataframe(self):
        """Return the matrix as a :class:`pandas.DataFrame` (genes x time)."""
        if _pd is None:  # pragma: no cover
            raise ImportError("pandas is required for to_dataframe().")
        return _pd.DataFrame(
            self.values, index=self.gene_names, columns=self.time_names
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ExpressionMatrix(n_genes={self.n_genes}, "
            f"n_time={self.n_time})"
        )


def _as_str_list(seq: Sequence, prefix: str, n: int) -> List[str]:
    if seq is None:
        return [f"{prefix}{i + 1}" for i in range(n)]
    return [str(s) for s in seq]


def as_expression_matrix(x) -> ExpressionMatrix:
    """Coerce ``x`` into an :class:`ExpressionMatrix`.

    Parameters
    ----------
    x : numpy.ndarray | pandas.DataFrame | anndata.AnnData | ExpressionMatrix
        A ``genes x timepoints`` expression table.

    Returns
    -------
    ExpressionMatrix
    """
    if isinstance(x, ExpressionMatrix):
        return x

    # AnnData ----------------------------------------------------------
    if _ad is not None and isinstance(x, _ad.AnnData):
        mat = x.X
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        mat = np.asarray(mat, dtype=np.float64)
        genes = _as_str_list(list(x.obs_names), "gene", mat.shape[0])
        times = _as_str_list(list(x.var_names), "t", mat.shape[1])
        return ExpressionMatrix(mat, genes, times)

    # pandas DataFrame -------------------------------------------------
    if _pd is not None and isinstance(x, _pd.DataFrame):
        mat = np.asarray(x.to_numpy(), dtype=np.float64)
        genes = _as_str_list(list(x.index), "gene", mat.shape[0])
        times = _as_str_list(list(x.columns), "t", mat.shape[1])
        return ExpressionMatrix(mat, genes, times)

    # numpy array (or array-like) --------------------------------------
    mat = np.asarray(x, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(
            f"Expression input must be 2-D (genes x timepoints); "
            f"got ndim={mat.ndim}."
        )
    genes = _as_str_list(None, "gene", mat.shape[0])
    times = _as_str_list(None, "t", mat.shape[1])
    return ExpressionMatrix(mat, genes, times)
