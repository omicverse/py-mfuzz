"""Example datasets bundled with pymfuzz.

Mfuzz ships ``data(yeast)`` -- the Cho et al. (1998) *Saccharomyces
cerevisiae* cell-cycle time-course (3000 genes x 17 timepoints).  When
the CMAP R environment with Mfuzz installed is available, :func:`load_yeast`
extracts that exact matrix so Python analyses run on identical input to R.
A small synthetic time-course is also provided for offline testing.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from .dataset import ExpressionMatrix

__all__ = ["load_yeast", "make_synthetic_timecourse"]

_CONDA_SH = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
_CMAP_ENV = "/scratch/users/steorra/env/CMAP"


def load_yeast(cache_dir: Optional[str] = None) -> ExpressionMatrix:
    """Load Mfuzz's ``data(yeast)`` cell-cycle time-course.

    Exports the dataset from the R Mfuzz package (via the CMAP conda
    environment) to a TSV the first time, then caches it.

    Parameters
    ----------
    cache_dir : str, optional
        Where to cache the exported TSV; defaults to a temp directory.

    Returns
    -------
    ExpressionMatrix
        The 3000 x 17 yeast expression matrix (with NA values).

    Raises
    ------
    RuntimeError
        If R / Mfuzz are not available and no cache exists.
    """
    cache = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir())
    cache.mkdir(parents=True, exist_ok=True)
    tsv = cache / "pymfuzz_yeast.tsv"

    if not tsv.exists():
        rscript = (
            "suppressMessages(library(Mfuzz)); data(yeast); "
            f"write.table(exprs(yeast), '{tsv}', sep='\\t', "
            "quote=FALSE, col.names=NA)"
        )
        cmd = (
            f"source {_CONDA_SH} && conda activate {_CMAP_ENV} "
            f"&& Rscript -e \"{rscript}\""
        )
        res = subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if res.returncode != 0 or not tsv.exists():
            raise RuntimeError(
                "Could not export data(yeast) from R Mfuzz.\n"
                + res.stderr[-1000:]
            )

    import pandas as pd

    df = pd.read_csv(tsv, sep="\t", index_col=0)
    return ExpressionMatrix(
        np.asarray(df.to_numpy(), dtype=np.float64),
        [str(g) for g in df.index],
        [str(t) for t in df.columns],
    )


def make_synthetic_timecourse(
    n_genes: int = 240,
    n_time: int = 12,
    n_patterns: int = 6,
    noise: float = 0.25,
    random_state: int = 0,
) -> ExpressionMatrix:
    """Generate a synthetic time-course with known cluster structure.

    Useful for offline smoke tests: ``n_patterns`` archetypal temporal
    profiles, each replicated and jittered with Gaussian noise.

    Parameters
    ----------
    n_genes : int, default 240
    n_time : int, default 12
    n_patterns : int, default 6
    noise : float, default 0.25
        Gaussian noise standard deviation.
    random_state : int, default 0

    Returns
    -------
    ExpressionMatrix
    """
    rng = np.random.default_rng(random_state)
    t = np.linspace(0, 2 * np.pi, n_time)
    archetypes = []
    for k in range(n_patterns):
        phase = 2 * np.pi * k / n_patterns
        archetypes.append(np.sin(t + phase) + 0.3 * np.cos(2 * t))
    archetypes = np.array(archetypes)

    rows = []
    names = []
    for g in range(n_genes):
        k = g % n_patterns
        rows.append(archetypes[k] + rng.normal(0, noise, n_time))
        names.append(f"gene{g + 1}")
    return ExpressionMatrix(
        np.array(rows, dtype=np.float64),
        names,
        [f"t{i + 1}" for i in range(n_time)],
    )
