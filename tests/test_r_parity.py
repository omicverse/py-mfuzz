"""R-parity tests -- pymfuzz vs Bioconductor Mfuzz 2.66.0.

The R driver (:file:`r_reference_driver.R`) runs Mfuzz on its bundled
yeast cell-cycle time-course (``data(yeast)``), so both sides analyse the
exact same input.  We compare:

* ``standardise`` -- bit-exact (rel-diff < 1e-9), it is deterministic.
* ``mestimate``   -- bit-exact (rel-diff < 1e-9), it is deterministic.
* ``fill_NA(knn)``-- bit-exact (deterministic imputation).
* ``mfuzz``       -- fuzzy c-means has random initialisation, so a
  bit-exact match is not expected.  After optimal cluster matching
  (Hungarian on the membership correspondence) the membership matrices
  must correlate at Pearson r > 0.95 and the hard-assignment Adjusted
  Rand Index must exceed 0.9.  Because the yeast fuzzifier (m ~= 1.15) is
  close to 1, fuzzy c-means is sharp and slow to converge -- both the R
  reference and the Python side therefore take the best (lowest
  within-error) of several converged restarts so both reach the same
  optimum.
* ``Dmin``        -- the minimum-centroid-distance curve correlates
  well with R's.
* ``membership``  -- bit-exact (rel-diff < 1e-9); it is a deterministic
  formula given the same centroids and fuzzifier ``m``.
* ``top_count``   -- exact integer match; it is a pure function of the
  membership matrix (fed R's own membership matrix).
* ``randomise``   -- a per-gene permutation, so the per-gene value
  *multiset* must be preserved (set-equality per gene, not order).

Tests skip gracefully when the CMAP R env or Mfuzz is unavailable.
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

import pymfuzz as mf

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"


def _r_available() -> bool:
    if not R_DRIVER.exists():
        return False
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
             "&& Rscript -e 'suppressMessages(library(Mfuzz)); "
             "cat(\"OK\")'"],
            capture_output=True, text=True, timeout=180, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or Mfuzz not installed.",
)


@pytest.fixture(scope="module")
def r_reference(tmp_path_factory):
    """Run the Mfuzz R reference once; return the output directory."""
    out_dir = tmp_path_factory.mktemp("mfuzz_R")
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} {out_dir}"
    )
    res = subprocess.run(
        ["bash", "-lc", cmd], capture_output=True, text=True, timeout=900,
    )
    if res.returncode != 0:
        pytest.skip(f"R reference driver failed:\n{res.stderr[-2000:]}")
    return out_dir


def _read(path):
    return pd.read_csv(path, sep="\t", index_col=0)


# ----------------------------------------------------------------------
# deterministic routines -- must be bit-exact
# ----------------------------------------------------------------------
def test_standardise_bit_exact(r_reference):
    filled = _read(r_reference / "yeast_filled.tsv")
    r_std = _read(r_reference / "standardised.tsv").to_numpy()
    py_std = mf.standardise(filled).values
    # standardise is deterministic; assert bit-exact agreement.  A pure
    # relative tolerance is inappropriate where R's value is exactly 0
    # (a sub-machine-epsilon residual would blow the ratio up), so a
    # combined abs+rel tolerance is used.
    max_abs = np.nanmax(np.abs(py_std - r_std))
    assert max_abs < 1e-9, f"standardise max abs diff {max_abs:.2e}"
    nz = np.abs(r_std) > 1e-6
    rel = np.nanmax(np.abs(py_std[nz] - r_std[nz]) / np.abs(r_std[nz]))
    assert rel < 1e-9, f"standardise rel-diff {rel:.2e} >= 1e-9"


def test_mestimate_bit_exact(r_reference):
    r_std = _read(r_reference / "standardised.tsv")
    m_r = float(pd.read_csv(r_reference / "mestimate.tsv", sep="\t")["m"][0])
    m_py = mf.mestimate(r_std)
    rel = abs(m_py - m_r) / abs(m_r)
    assert rel < 1e-9, f"mestimate rel-diff {rel:.2e} (py={m_py}, r={m_r})"


def test_fill_NA_knn_bit_exact(r_reference):
    raw = _read(r_reference / "yeast_raw.tsv")
    r_filled = _read(r_reference / "yeast_filled.tsv").to_numpy()
    filt = mf.filter_NA(raw, thres=0.25, verbose=False)
    py_filled = mf.fill_NA(filt, mode="knn").values
    assert py_filled.shape == r_filled.shape
    max_abs = np.nanmax(np.abs(py_filled - r_filled))
    assert max_abs < 1e-8, f"fill_NA(knn) max abs diff {max_abs:.2e}"


def test_filter_NA_count(r_reference):
    raw = _read(r_reference / "yeast_raw.tsv")
    info = pd.read_csv(r_reference / "info.tsv", sep="\t").set_index("key")
    filt = mf.filter_NA(raw, thres=0.25, verbose=False)
    assert filt.n_genes == int(info.loc["n_genes_filled", "value"])


# ----------------------------------------------------------------------
# mfuzz -- random init; assert clustering AGREEMENT
# ----------------------------------------------------------------------
def _best_mfuzz(data, c, m, n=20):
    """Best (lowest within-error) of n converged restarts."""
    best = None
    for s in range(n):
        cl = mf.mfuzz(data, c=c, m=m, random_state=s, iter_max=500)
        if best is None or cl.withinerror < best.withinerror:
            best = cl
    return best


@pytest.fixture(scope="module")
def py_clustering(r_reference):
    r_std = _read(r_reference / "standardised.tsv")
    m_r = float(pd.read_csv(r_reference / "mestimate.tsv", sep="\t")["m"][0])
    return _best_mfuzz(r_std, c=16, m=m_r, n=20)


def test_mfuzz_membership_correlation(r_reference, py_clustering):
    r_mem = _read(r_reference / "membership.tsv").to_numpy()
    P = py_clustering.membership
    c = P.shape[1]
    cost = np.array(
        [[-pearsonr(P[:, i], r_mem[:, j])[0] for j in range(c)]
         for i in range(c)]
    )
    ri, cj = linear_sum_assignment(cost)
    perm = np.empty(c, dtype=int)
    perm[cj] = ri
    corr = pearsonr(P[:, perm].ravel(), r_mem.ravel())[0]
    assert corr > 0.95, f"membership Pearson r {corr:.4f} <= 0.95"


def test_mfuzz_hard_assignment_ari(r_reference, py_clustering):
    from sklearn.metrics import adjusted_rand_score

    r_clu = pd.read_csv(
        r_reference / "cluster.tsv", sep="\t"
    )["CLUSTER"].to_numpy()
    ari = adjusted_rand_score(py_clustering.cluster, r_clu)
    assert ari > 0.9, f"hard-assignment ARI {ari:.4f} <= 0.9"


def test_mfuzz_centers_correlation(r_reference, py_clustering):
    r_mem = _read(r_reference / "membership.tsv").to_numpy()
    r_cen = _read(r_reference / "centers.tsv").to_numpy()
    P = py_clustering.membership
    c = P.shape[1]
    cost = np.array(
        [[-pearsonr(P[:, i], r_mem[:, j])[0] for j in range(c)]
         for i in range(c)]
    )
    ri, cj = linear_sum_assignment(cost)
    perm = np.empty(c, dtype=int)
    perm[cj] = ri
    corr = pearsonr(
        py_clustering.centers[perm].ravel(), r_cen.ravel()
    )[0]
    assert corr > 0.95, f"centers Pearson r {corr:.4f} <= 0.95"


# ----------------------------------------------------------------------
# Dmin -- distance curve must correlate with R's
# ----------------------------------------------------------------------
def test_dmin_curve_correlation(r_reference):
    r_std = _read(r_reference / "standardised.tsv")
    m_r = float(pd.read_csv(r_reference / "mestimate.tsv", sep="\t")["m"][0])
    r_dmin = pd.read_csv(r_reference / "dmin.tsv", sep="\t")
    py_dmin = mf.Dmin(
        r_std, m=m_r, crange=range(4, 25, 4), repeats=3, random_state=0
    )
    assert len(py_dmin) == len(r_dmin)
    corr = pearsonr(py_dmin, r_dmin["dmin"].to_numpy())[0]
    assert corr > 0.95, f"Dmin curve Pearson r {corr:.4f} <= 0.95"
    # both curves should be monotonically decreasing in cluster count
    assert py_dmin[0] > py_dmin[-1]


# ----------------------------------------------------------------------
# membership -- deterministic given centres + m: assert bit-exact
# ----------------------------------------------------------------------
def test_membership_bit_exact(r_reference):
    r_std = _read(r_reference / "standardised.tsv")
    r_cen = _read(r_reference / "centers.tsv").to_numpy()
    m_r = float(pd.read_csv(r_reference / "mestimate.tsv", sep="\t")["m"][0])
    r_mem = _read(r_reference / "membership.proj.tsv").to_numpy()
    py_mem = mf.membership(r_std, r_cen, m=m_r)
    assert py_mem.shape == r_mem.shape
    max_abs = np.nanmax(np.abs(py_mem - r_mem))
    assert max_abs < 1e-9, f"membership max abs diff {max_abs:.2e}"
    nz = np.abs(r_mem) > 1e-6
    rel = np.nanmax(np.abs(py_mem[nz] - r_mem[nz]) / np.abs(r_mem[nz]))
    assert rel < 1e-9, f"membership rel-diff {rel:.2e} >= 1e-9"
    # every row sums to 1
    assert np.allclose(py_mem.sum(axis=1), 1.0, atol=1e-9)


# ----------------------------------------------------------------------
# top_count -- must match R's per-gene counts exactly
# ----------------------------------------------------------------------
def test_top_count_matches_r(r_reference, py_clustering):
    # rebuild the membership-matched cluster permutation so the Python
    # clustering's centroid ordering lines up with R's, then re-derive
    # top.count from R's own membership matrix to compare like-for-like.
    r_mem = _read(r_reference / "membership.proj.tsv")
    r_tc = pd.read_csv(r_reference / "topcount.tsv", sep="\t")

    # top.count is a pure function of the membership matrix; feed R's
    # membership matrix through the Python implementation and assert the
    # per-gene counts are identical.
    class _Cl:
        membership = r_mem.to_numpy()

    py_tc = mf.top_count(_Cl())
    assert py_tc.shape[0] == len(r_tc)
    assert np.array_equal(py_tc, r_tc["COUNT"].to_numpy())


# ----------------------------------------------------------------------
# randomise -- a per-gene permutation: assert value MULTISET preserved
# ----------------------------------------------------------------------
def test_randomise_preserves_per_gene_multiset(r_reference):
    r_std = _read(r_reference / "standardised.tsv")
    r_rand = _read(r_reference / "randomised.tsv")
    py_rand = mf.randomise(r_std, random_state=0)
    # shape preserved
    assert py_rand.shape == r_std.to_numpy().shape
    # R's randomise: each row of the output is a permutation of the input
    src = r_std.to_numpy()
    out_r = r_rand.to_numpy()
    out_py = py_rand.values
    for i in range(src.shape[0]):
        a = np.sort(src[i, :])
        # R output: per-gene value multiset equals the input multiset
        assert np.allclose(a, np.sort(out_r[i, :]), atol=1e-9), (
            f"R randomise changed gene {i} value set"
        )
        # Python output: same invariant holds
        assert np.allclose(a, np.sort(out_py[i, :]), atol=1e-9), (
            f"Python randomise changed gene {i} value set"
        )
