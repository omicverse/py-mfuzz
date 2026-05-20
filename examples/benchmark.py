"""Benchmark pymfuzz on the Mfuzz yeast cell-cycle time-course.

Runs the full Mfuzz workflow -- filter / fill / standardise / mestimate /
fuzzy c-means / acore / Dmin -- and reports timings.  Falls back to a
synthetic time-course when the R Mfuzz yeast dataset is unavailable.

Usage
-----
    /scratch/users/steorra/env/omicdev/bin/python examples/benchmark.py
"""
from __future__ import annotations

import time

import numpy as np

import pymfuzz as mf


def _timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"  {label:<34s} {dt * 1e3:9.2f} ms")
    return out


def main() -> None:
    print("pymfuzz benchmark")
    print("=" * 56)

    try:
        data = mf.load_yeast()
        print(f"dataset: Mfuzz yeast  ({data.n_genes} genes x "
              f"{data.n_time} timepoints)")
    except Exception as exc:  # pragma: no cover
        print(f"(yeast unavailable: {exc}; using synthetic)")
        data = mf.make_synthetic_timecourse(
            n_genes=3000, n_time=17, n_patterns=16, random_state=0
        )
        print(f"dataset: synthetic    ({data.n_genes} genes x "
              f"{data.n_time} timepoints)")
    print("-" * 56)

    filt = _timed("filter_NA(thres=0.25)",
                  lambda: mf.filter_NA(data, thres=0.25, verbose=False))
    filled = _timed("fill_NA(mode='knn')",
                    lambda: mf.fill_NA(filt, mode="knn"))
    std = _timed("standardise", lambda: mf.standardise(filled))
    m = _timed("mestimate", lambda: mf.mestimate(std))
    print(f"  -> estimated fuzzifier m = {m:.6f}")

    cl = _timed("mfuzz(c=16)",
                lambda: mf.mfuzz(std, c=16, m=m, random_state=0,
                                 iter_max=500))
    print(f"  -> {cl.iter} iterations, within-error = "
          f"{cl.withinerror:.4f}")

    cores = _timed("acore(min_acore=0.5)",
                   lambda: mf.acore(std, cl, min_acore=0.5))
    core_sizes = [len(c) for c in cores]
    print(f"  -> core sizes: min={min(core_sizes)}, "
          f"max={max(core_sizes)}, total={sum(core_sizes)}")

    _timed("overlap", lambda: mf.overlap(cl))
    _timed("Dmin(crange=4..20)",
           lambda: mf.Dmin(std, m=m, crange=range(4, 21, 4),
                           repeats=2, random_state=0))
    print("=" * 56)
    print("done.")


if __name__ == "__main__":
    main()
