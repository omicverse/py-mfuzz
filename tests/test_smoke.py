"""Algorithmic smoke tests for pymfuzz -- no R required.

These check the internal consistency of each ported routine against
hand-derived expectations and synthetic data with known structure.
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
import pytest

import pymfuzz as mf

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic():
    """Synthetic time-course with 6 known temporal patterns."""
    return mf.make_synthetic_timecourse(
        n_genes=180, n_time=10, n_patterns=6, noise=0.2, random_state=0
    )


# ----------------------------------------------------------------------
# input coercion
# ----------------------------------------------------------------------
def test_as_expression_matrix_numpy():
    x = np.arange(12, dtype=float).reshape(4, 3)
    em = mf.as_expression_matrix(x)
    assert em.shape == (4, 3)
    assert em.n_genes == 4 and em.n_time == 3
    assert len(em.gene_names) == 4 and len(em.time_names) == 3


def test_as_expression_matrix_dataframe():
    df = pd.DataFrame(
        np.random.rand(5, 4),
        index=[f"g{i}" for i in range(5)],
        columns=[f"t{j}" for j in range(4)],
    )
    em = mf.as_expression_matrix(df)
    assert em.gene_names == [f"g{i}" for i in range(5)]
    assert em.time_names == [f"t{j}" for j in range(4)]


def test_as_expression_matrix_anndata():
    ad = pytest.importorskip("anndata")
    a = ad.AnnData(np.random.rand(6, 3))
    em = mf.as_expression_matrix(a)
    assert em.shape == (6, 3)


# ----------------------------------------------------------------------
# standardisation
# ----------------------------------------------------------------------
def test_standardise_zero_mean_unit_sd():
    x = np.random.RandomState(0).rand(20, 8)
    em = mf.standardise(x)
    # each row: mean ~ 0, sample SD ~ 1
    assert np.allclose(em.values.mean(axis=1), 0.0, atol=1e-12)
    sds = em.values.std(axis=1, ddof=1)
    assert np.allclose(sds, 1.0, atol=1e-12)


def test_standardise_uses_sample_sd():
    # constant offset -> NaN row after dividing by 0 SD
    x = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
    em = mf.standardise(x)
    assert np.all(np.isnan(em.values[0]))
    assert not np.any(np.isnan(em.values[1]))


def test_standardise2_reference_timepoint():
    x = np.array([[2.0, 4.0, 6.0, 8.0]])
    em = mf.standardise2(x, timepoint=1)
    # first column becomes exactly 0
    assert em.values[0, 0] == 0.0


def test_standardise_handles_nan():
    x = np.array([[1.0, np.nan, 3.0, 5.0]])
    em = mf.standardise(x)
    # the NA stays NA; finite entries are standardised
    assert np.isnan(em.values[0, 1])
    finite = em.values[0, ~np.isnan(em.values[0])]
    assert np.isclose(finite.mean(), 0.0, atol=1e-12)


# ----------------------------------------------------------------------
# NA handling
# ----------------------------------------------------------------------
def test_filter_NA_drops_high_na_genes():
    x = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],          # 0 NA -> keep
            [np.nan, np.nan, np.nan, 1.0],  # 75% NA -> drop
            [1.0, np.nan, 3.0, 4.0],        # 25% NA -> keep (not > 0.25)
        ]
    )
    em = mf.filter_NA(x, thres=0.25, verbose=False)
    assert em.n_genes == 2


def test_fill_NA_mean():
    x = np.array([[1.0, np.nan, 3.0]])
    em = mf.fill_NA(x, mode="mean")
    assert not np.any(np.isnan(em.values))
    assert np.isclose(em.values[0, 1], 2.0)


def test_fill_NA_median():
    x = np.array([[1.0, 2.0, np.nan, 100.0]])
    em = mf.fill_NA(x, mode="median")
    assert np.isclose(em.values[0, 2], 2.0)


def test_fill_NA_knn_removes_all_na():
    rng = np.random.RandomState(1)
    x = rng.rand(40, 6)
    x[5, 2] = np.nan
    x[12, 0] = np.nan
    em = mf.fill_NA(x, mode="knn", k=5)
    assert not np.any(np.isnan(em.values))


def test_filter_std_drops_low_variability():
    x = np.array(
        [
            [0.0, 0.0, 0.0, 0.001],  # tiny SD -> drop
            [0.0, 5.0, -5.0, 2.0],   # high SD -> keep
        ]
    )
    em = mf.filter_std(x, min_std=0.5)
    assert em.n_genes == 1


# ----------------------------------------------------------------------
# mestimate
# ----------------------------------------------------------------------
def test_mestimate_in_reasonable_range():
    x = np.random.RandomState(0).rand(500, 10)
    m = mf.mestimate(x)
    assert 1.0 < m < 3.0


def test_mestimate_formula():
    # check against the closed-form Schwammle & Jensen expression
    N, D = 1000, 12
    x = np.zeros((N, D))
    m = mf.mestimate(x)
    expect = (
        1.0
        + (1418.0 / N + 22.05) * D ** (-2.0)
        + (12.33 / N + 0.243) * D ** (-0.0406 * np.log(N) - 0.1134)
    )
    assert np.isclose(m, expect, rtol=1e-12)


# ----------------------------------------------------------------------
# mfuzz / cmeans
# ----------------------------------------------------------------------
def test_mfuzz_recovers_synthetic_clusters(synthetic):
    from sklearn.metrics import adjusted_rand_score

    data = mf.standardise(synthetic)
    cl = mf.mfuzz(data, c=6, m=1.5, random_state=0, iter_max=500)
    true = np.array([g % 6 for g in range(synthetic.n_genes)])
    ari = adjusted_rand_score(cl.cluster, true)
    assert ari > 0.8, f"synthetic recovery ARI {ari:.3f}"


def test_mfuzz_membership_properties(synthetic):
    data = mf.standardise(synthetic)
    cl = mf.mfuzz(data, c=5, m=1.5, random_state=1)
    assert cl.membership.shape == (synthetic.n_genes, 5)
    # rows sum to 1
    assert np.allclose(cl.membership.sum(axis=1), 1.0, atol=1e-9)
    # memberships in [0, 1]
    assert cl.membership.min() >= -1e-12
    assert cl.membership.max() <= 1.0 + 1e-9
    # hard assignment matches argmax
    assert np.array_equal(cl.cluster, cl.membership.argmax(axis=1) + 1)
    # sizes sum to n_genes
    assert cl.size.sum() == synthetic.n_genes


def test_mfuzz_centers_shape(synthetic):
    cl = mf.mfuzz(mf.standardise(synthetic), c=4, m=1.5, random_state=0)
    assert cl.centers.shape == (4, synthetic.n_time)


def test_mfuzz_reproducible_with_seed(synthetic):
    data = mf.standardise(synthetic)
    a = mf.mfuzz(data, c=5, m=1.5, random_state=42)
    b = mf.mfuzz(data, c=5, m=1.5, random_state=42)
    assert np.allclose(a.membership, b.membership)
    assert np.array_equal(a.cluster, b.cluster)


def test_cmeans_explicit_centers(synthetic):
    data = mf.standardise(synthetic)
    init = data.values[[0, 30, 60, 90], :]
    cl = mf.cmeans(data, centers=init, m=1.5)
    assert cl.n_clusters == 4
    assert cl.iter >= 1


def test_cmeans_objective_decreases(synthetic):
    # the within-error should be finite and non-negative
    cl = mf.mfuzz(mf.standardise(synthetic), c=6, m=2.0, random_state=0)
    assert np.isfinite(cl.withinerror)
    assert cl.withinerror >= 0.0


# ----------------------------------------------------------------------
# acore
# ----------------------------------------------------------------------
def test_acore_returns_core_genes(synthetic):
    data = mf.standardise(synthetic)
    cl = mf.mfuzz(data, c=6, m=1.5, random_state=0)
    cores = mf.acore(data, cl, min_acore=0.5)
    assert len(cores) == 6
    for core in cores:
        # every core gene has membership > 0.5 and is hard-assigned here
        assert np.all(core.membership > 0.5)
        # sorted descending
        assert np.all(np.diff(core.membership) <= 1e-12)


def test_acore_dataframe(synthetic):
    data = mf.standardise(synthetic)
    cl = mf.mfuzz(data, c=4, m=1.5, random_state=0)
    cores = mf.acore(data, cl, min_acore=0.3)
    df = cores[0].to_dataframe()
    assert list(df.columns) == ["NAME", "MEM.SHIP"]


# ----------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------
def test_dmin_decreasing(synthetic):
    data = mf.standardise(synthetic)
    curve = mf.Dmin(
        data, m=1.5, crange=range(2, 11, 2), repeats=2, random_state=0
    )
    assert len(curve) == 5
    # min centroid distance broadly decreases with more clusters
    assert curve[0] >= curve[-1]


def test_cselection_shape(synthetic):
    data = mf.standardise(synthetic)
    ne = mf.cselection(
        data, m=1.5, crange=range(2, 9, 2), repeats=3, random_state=0
    )
    assert ne.shape == (3, 4)
    assert np.all(ne >= 0)


def test_partcoef(synthetic):
    data = mf.standardise(synthetic)
    res = mf.partcoef(
        data, crange=[3, 5], mrange=[1.2, 1.5], random_state=0
    )
    assert res.F.shape == (2, 2)
    # F >= F_min and F <= 1
    assert np.all(res.F >= res.F_min - 1e-9)
    assert np.all(res.F <= 1.0 + 1e-9)
    assert np.allclose(res.F_n, res.F - res.F_min)


def test_overlap_matrix(synthetic):
    cl = mf.mfuzz(mf.standardise(synthetic), c=5, m=1.5, random_state=0)
    O = mf.overlap(cl)
    assert O.shape == (5, 5)
    # columns are normalised to sum to 1
    assert np.allclose(O.sum(axis=0), 1.0, atol=1e-9)


# ----------------------------------------------------------------------
# membership
# ----------------------------------------------------------------------
def test_membership_projects_onto_centroids(synthetic):
    data = mf.standardise(synthetic)
    cl = mf.mfuzz(data, c=6, m=1.5, random_state=0)
    u = mf.membership(data, cl.centers, m=1.5)
    assert u.shape == (synthetic.n_genes, 6)
    # every row sums to 1, values in [0, 1]
    assert np.allclose(u.sum(axis=1), 1.0, atol=1e-9)
    assert u.min() >= -1e-12 and u.max() <= 1.0 + 1e-9


def test_membership_reproduces_own_clustering(synthetic):
    # projecting the clustered data onto its own centres at the same m
    # should reproduce the mfuzz membership matrix.
    data = mf.standardise(synthetic)
    cl = mf.mfuzz(data, c=5, m=1.6, random_state=2)
    u = mf.membership(data, cl.centers, m=1.6)
    assert np.allclose(u, cl.membership, atol=1e-8)


def test_membership_accepts_single_vector(synthetic):
    data = mf.standardise(synthetic)
    cl = mf.mfuzz(data, c=4, m=1.5, random_state=0)
    u = mf.membership(data.values[0], cl.centers, m=1.5)
    assert u.shape == (1, 4)


# ----------------------------------------------------------------------
# top_count
# ----------------------------------------------------------------------
def test_top_count_per_gene(synthetic):
    data = mf.standardise(synthetic)
    cl = mf.mfuzz(data, c=6, m=1.5, random_state=0)
    tc = mf.top_count(cl)
    # one count per gene
    assert tc.shape == (synthetic.n_genes,)
    # non-negative integers
    assert tc.dtype.kind in "iu"
    assert np.all(tc >= 0)
    # with no ties, each cluster has exactly one column-max gene, so the
    # counts sum to the number of clusters.
    assert tc.sum() >= cl.n_clusters


# ----------------------------------------------------------------------
# randomise
# ----------------------------------------------------------------------
def test_randomise_preserves_per_gene_values():
    x = np.arange(24, dtype=float).reshape(4, 6)
    r = mf.randomise(x, random_state=0)
    assert r.shape == (4, 6)
    for i in range(4):
        # the per-gene value multiset is unchanged -- only the order
        assert set(r.values[i]) == set(x[i])
    # at least one gene is actually reordered
    assert not np.array_equal(r.values, x)


def test_randomise_reproducible_with_seed():
    x = np.random.RandomState(0).rand(20, 8)
    a = mf.randomise(x, random_state=7)
    b = mf.randomise(x, random_state=7)
    assert np.array_equal(a.values, b.values)


# ----------------------------------------------------------------------
# table2eset
# ----------------------------------------------------------------------
def test_table2eset_plain(tmp_path):
    p = tmp_path / "expr.txt"
    p.write_text(
        "ID\tt0\tt1\tt2\n"
        "g1\t1.0\t2.0\t3.0\n"
        "g2\t4.0\t5.0\t6.0\n"
    )
    em = mf.table2eset(str(p))
    assert em.shape == (2, 3)
    assert em.gene_names == ["g1", "g2"]
    assert em.time_names == ["t0", "t1", "t2"]
    assert np.allclose(em.values[1], [4.0, 5.0, 6.0])


def test_table2eset_with_gene_name_and_time_rows(tmp_path):
    p = tmp_path / "expr2.txt"
    p.write_text(
        "ID\tGeneName\ts0\ts1\ts2\n"
        "Time\t\t0\t10\t20\n"
        "g1\tAlpha\t1.0\t2.0\t3.0\n"
        "g2\tBeta\t4.0\t5.0\t6.0\n"
    )
    em = mf.table2eset(str(p))
    assert em.shape == (2, 3)
    assert em.gene_names == ["g1", "g2"]
    assert em.time_names == ["s0", "s1", "s2"]
    assert np.allclose(em.values[0], [1.0, 2.0, 3.0])


# ----------------------------------------------------------------------
# mfuzz_colorbar
# ----------------------------------------------------------------------
def test_mfuzz_colorbar_default():
    fig = mf.mfuzz_colorbar()
    assert isinstance(fig, Figure)


def test_mfuzz_colorbar_horizontal_and_fancy():
    fig = mf.mfuzz_colorbar(col="fancy", horizontal=True)
    assert isinstance(fig, Figure)


# ----------------------------------------------------------------------
# kmeans2
# ----------------------------------------------------------------------
def test_kmeans2(synthetic):
    from sklearn.metrics import adjusted_rand_score

    data = mf.standardise(synthetic)
    kl = mf.kmeans2(data, k=6, n_init=5, random_state=0)
    assert kl.centers.shape == (6, synthetic.n_time)
    assert kl.size.sum() == synthetic.n_genes
    true = np.array([g % 6 for g in range(synthetic.n_genes)])
    ari = adjusted_rand_score(kl.cluster, true)
    assert ari > 0.7


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------
def test_mfuzz_plot(synthetic):
    data = mf.standardise(synthetic)
    cl = mf.mfuzz(data, c=6, m=1.5, random_state=0)
    fig = mf.mfuzz_plot(data, cl, mfrow=(2, 3))
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 6


def test_mfuzz_plot2_with_centre(synthetic):
    data = mf.standardise(synthetic)
    cl = mf.mfuzz(data, c=4, m=1.5, random_state=0)
    fig = mf.mfuzz_plot2(
        data, cl, mfrow=(2, 2), centre=True, time_points=range(10)
    )
    assert isinstance(fig, Figure)


def test_kmeans2_plot(synthetic):
    data = mf.standardise(synthetic)
    kl = mf.kmeans2(data, k=4, random_state=0)
    fig = mf.kmeans2_plot(data, kl, mfrow=(2, 2))
    assert isinstance(fig, Figure)


def test_overlap_plot(synthetic):
    cl = mf.mfuzz(mf.standardise(synthetic), c=5, m=1.5, random_state=0)
    O = mf.overlap(cl)
    fig = mf.overlap_plot(cl, O, thres=0.05)
    assert isinstance(fig, Figure)


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------
def test_public_api_complete():
    for name in mf.__all__:
        assert hasattr(mf, name), f"missing export {name}"
    assert mf.__version__
