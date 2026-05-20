# py-mfuzz

**Pure-Python port of the Bioconductor package
[Mfuzz](https://bioconductor.org/packages/release/bioc/html/Mfuzz.html)**
— soft clustering of time-series gene-expression data by fuzzy c-means
(Futschik & Carlisle, *J. Bioinform. Comput. Biol.* 2005; Kumar &
Futschik, *Bioinformation* 2007).

`pymfuzz` reproduces the full computational and visualisation API of
Mfuzz with **no R dependency** — only `numpy`, `scipy`, `pandas`,
`matplotlib` and `anndata`. The fuzzy c-means core is a faithful numpy
port of `e1071`'s `cmeans` C routine, the same algorithm R's `mfuzz()`
wraps.

## Why

Mfuzz operates on a Bioconductor `ExpressionSet`. `pymfuzz` instead
accepts a plain **genes × timepoints** `numpy.ndarray`,
`pandas.DataFrame` or `anndata.AnnData`, and returns numpy / pandas /
dataclasses — drop-in for Python single-cell / bulk pipelines.

## Install

```bash
pip install pymfuzz
```

From source:

```bash
pip install -e .
```

## Quick start

```python
import pymfuzz as mf

# 1. load a genes x timepoints time-course (Mfuzz's data(yeast))
data = mf.load_yeast()

# 2. preprocessing
data = mf.filter_NA(data, thres=0.25)   # drop genes with many NAs
data = mf.fill_NA(data, mode="knn")     # impute remaining NAs
data = mf.standardise(data)             # per-gene z-score

# 3. estimate the fuzzifier and cluster
m  = mf.mestimate(data)                 # Schwammle & Jensen (2010)
cl = mf.mfuzz(data, c=16, m=m, random_state=0)

# 4. extract core genes and plot
cores = mf.acore(data, cl, min_acore=0.5)
fig   = mf.mfuzz_plot(data, cl, mfrow=(4, 4))
```

## API

| Group | Functions |
|-------|-----------|
| **Data structures** | `ExpressionMatrix`, `as_expression_matrix`, `FClust`, `KMeansResult`, `AcoreCluster`, `PartcoefResult` |
| **Preprocessing** | `standardise`, `standardise2`, `filter_NA`, `fill_NA`, `filter_std` |
| **Clustering** | `mestimate`, `mfuzz`, `cmeans` |
| **Diagnostics** | `acore`, `Dmin`, `cselection`, `partcoef`, `overlap` |
| **Hard clustering** | `kmeans2` |
| **Plotting** | `mfuzz_plot`, `mfuzz_plot2`, `kmeans2_plot`, `overlap_plot` |
| **Datasets** | `load_yeast`, `make_synthetic_timecourse` |

### Mapping to the R package

| Mfuzz (R) | pymfuzz (Python) |
|-----------|------------------|
| `standardise` / `standardise2` | `standardise` / `standardise2` |
| `mestimate` | `mestimate` |
| `mfuzz` (`e1071::cmeans`) | `mfuzz` / `cmeans` |
| `acore` | `acore` |
| `Dmin`, `cselection`, `partcoef` | `Dmin`, `cselection`, `partcoef` |
| `filter.NA`, `fill.NA`, `filter.std` | `filter_NA`, `fill_NA`, `filter_std` |
| `overlap`, `overlap.plot` | `overlap`, `overlap_plot` |
| `mfuzz.plot`, `mfuzz.plot2` | `mfuzz_plot`, `mfuzz_plot2` |
| `kmeans2`, `kmeans2.plot` | `kmeans2`, `kmeans2_plot` |

## R parity

Validated against **Mfuzz 2.66.0** / **e1071 1.7.17** on the bundled
yeast cell-cycle time-course (`data(yeast)`):

| Routine | Agreement vs R |
|---------|----------------|
| `standardise` | bit-exact (rel-diff ≈ 1e-15) |
| `mestimate`   | bit-exact (rel-diff ≈ 1e-15) |
| `fill_NA(knn)`| bit-exact (max abs diff ≈ 1e-15) |
| `mfuzz`       | membership Pearson r = 1.0, centres r = 1.0, hard-assignment ARI ≈ 0.99 |
| `Dmin`        | curve Pearson r = 1.0 |

`standardise`, `mestimate` and `fill_NA` are deterministic and match R to
machine precision. Fuzzy c-means uses **random initialisation**, so a
bit-exact match across RNGs is not expected; instead clustering
*agreement* is asserted (Hungarian-matched membership correlation and
Adjusted Rand Index). Because the yeast fuzzifier (`m ≈ 1.15`) is close
to 1, fuzzy c-means is sharp and slow to converge — both sides take the
best of several converged restarts so they reach the same optimum.

Run the parity tests (needs the CMAP R environment):

```bash
python -m pytest tests/ -q
```

## License

GPL-2 — the same license as the original Bioconductor Mfuzz package.
See [`LICENSE`](LICENSE).

## Citation

If you use `pymfuzz`, please cite the original Mfuzz papers:

- L. Kumar, M. Futschik (2007). *Mfuzz: a software package for soft
  clustering of microarray data.* Bioinformation 2(1):5–7.
- M. Futschik, B. Carlisle (2005). *Noise-robust soft clustering of gene
  expression time-course data.* J. Bioinform. Comput. Biol. 3(4):965–988.
