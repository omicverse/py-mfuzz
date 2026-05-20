"""pymfuzz: Pure-Python port of the Bioconductor package Mfuzz.

A standalone, dependency-light port of **Mfuzz** -- soft clustering of
time-series gene-expression data by fuzzy c-means (Futschik & Carlisle,
*J. Bioinform. Comput. Biol.* 2005; Kumar & Futschik, *Bioinformation*
2007).  It covers the full computational + visualisation API of Mfuzz:
standardisation, fuzzifier estimation, fuzzy c-means clustering,
core-gene extraction, cluster-number diagnostics and the signature
soft-cluster temporal-profile plots.

Where Mfuzz operates on a Bioconductor ``ExpressionSet``, pymfuzz accepts
a plain ``genes x timepoints`` :class:`numpy.ndarray`,
:class:`pandas.DataFrame` or :class:`anndata.AnnData`, and returns
numpy / pandas / dataclasses.

Reused engines
--------------
* The fuzzy c-means core is a faithful numpy port of ``e1071``'s
  ``cmeans`` C routine -- the same algorithm R's :func:`mfuzz` wraps.

Core data structures
--------------------
* :class:`ExpressionMatrix` -- the Python ``ExpressionSet`` (a
  ``genes x timepoints`` matrix with row / column labels).
* :class:`FClust` -- a fuzzy-clustering result (centres, membership,
  hard assignment, size, iterations, objective value).

Preprocessing
-------------
* :func:`standardise`  -- per-gene z-score (the Mfuzz preprocessing).
* :func:`standardise2` -- standardise relative to one timepoint.
* :func:`filter_NA`    -- drop genes with too many missing values.
* :func:`fill_NA`      -- impute missing values (mean / median / knn).
* :func:`filter_std`   -- drop low-variability genes.

Fuzzifier / clustering
----------------------
* :func:`mestimate` -- Schwammle & Jensen (2010) fuzzifier estimate.
* :func:`mfuzz`     -- fuzzy c-means soft clustering.
* :func:`cmeans`    -- the underlying fuzzy c-means engine.

Post-processing / diagnostics
-----------------------------
* :func:`acore`      -- per-cluster "core" genes.
* :func:`Dmin`       -- minimum-centroid-distance elbow diagnostic.
* :func:`cselection` -- empty-cluster frequency vs cluster count.
* :func:`partcoef`   -- partition coefficient over a ``(c, m)`` grid.
* :func:`overlap`    -- cluster-to-cluster membership overlap.

Hard clustering
---------------
* :func:`kmeans2` -- the k-means hard-clustering alternative.

Plotting
--------
* :func:`mfuzz_plot` / :func:`mfuzz_plot2` -- soft-cluster
  temporal-profile grids.
* :func:`kmeans2_plot` -- hard-cluster profile grid.
* :func:`overlap_plot` -- PCA overlap map of cluster centres.

Datasets
--------
* :func:`load_yeast` -- the Cho et al. yeast cell-cycle time-course
  (Mfuzz's ``data(yeast)``).
* :func:`make_synthetic_timecourse` -- a synthetic time-course.

Quick-start
-----------
>>> import pymfuzz as mf
>>> data = mf.load_yeast()                 # genes x timepoints
>>> data = mf.filter_NA(data, thres=0.25)
>>> data = mf.fill_NA(data, mode="knn")
>>> data = mf.standardise(data)
>>> m = mf.mestimate(data)
>>> cl = mf.mfuzz(data, c=16, m=m, random_state=0)
>>> cores = mf.acore(data, cl, min_acore=0.5)
>>> fig = mf.mfuzz_plot(data, cl, mfrow=(4, 4))
"""
from __future__ import annotations

from .analysis import (
    AcoreCluster,
    Dmin,
    PartcoefResult,
    acore,
    cselection,
    mestimate,
    overlap,
    partcoef,
)
from .cluster import FClust, cmeans, mfuzz
from .dataset import ExpressionMatrix, as_expression_matrix
from .datasets import load_yeast, make_synthetic_timecourse
from .kmeans import KMeansResult, kmeans2
from .plotting import kmeans2_plot, mfuzz_plot, mfuzz_plot2, overlap_plot
from .preprocess import (
    fill_NA,
    filter_NA,
    filter_std,
    standardise,
    standardise2,
)

__version__ = "0.1.0"

__all__ = [
    # data structures
    "ExpressionMatrix",
    "as_expression_matrix",
    "FClust",
    "KMeansResult",
    "AcoreCluster",
    "PartcoefResult",
    # preprocessing
    "standardise",
    "standardise2",
    "filter_NA",
    "fill_NA",
    "filter_std",
    # fuzzifier / clustering
    "mestimate",
    "mfuzz",
    "cmeans",
    # post-processing / diagnostics
    "acore",
    "Dmin",
    "cselection",
    "partcoef",
    "overlap",
    # hard clustering
    "kmeans2",
    # plotting
    "mfuzz_plot",
    "mfuzz_plot2",
    "kmeans2_plot",
    "overlap_plot",
    # datasets
    "load_yeast",
    "make_synthetic_timecourse",
]
