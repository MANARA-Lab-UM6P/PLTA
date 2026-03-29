#!/usr/bin/env python3
"""
data/compute_zeta.py — Compute the practical noise threshold ζ (Section VI-B).

From the paper:
    ζ = min_{i,j,k,l ∈ [N]}  H_{ijkl}

where H_{ijkl} is defined in Lemma 2.  Because ζ depends on the exact
locations of all users — which must remain private — we estimate it by
running many simulations on the Gowalla dataset and taking the minimum.

The paper ran 10,000 simulations with 33 tasks + 67 workers each, and
found:  min_{s ≤ 10,000} ζ_s = 0.0005

Usage (from the PLTA/ root):
    python data/compute_zeta.py [--simulations N]
"""

import argparse
import math
import pickle
import sys
import os
from random import sample

import numpy as np

# Allow running from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data_loader import load_gowalla


# ── Core geometry helpers ─────────────────────────────────────────────────────

def _euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _all_task_worker_distances(tasks, workers):
    """Return a list of ((task_idx, worker_idx), distance) sorted by distance."""
    pairs = [
        ((i, j), _euclidean(tasks[i], workers[j]))
        for i in range(len(tasks))
        for j in range(len(workers))
    ]
    pairs.sort(key=lambda x: x[1])
    return pairs


def _h_ijkl(pair_lo, pair_hi, tasks, workers):
    """
    Compute H_{ijkl} from Lemma 2 for two consecutive task-worker pairs.

    pair_lo  — ((task_i, worker_i), d_lo)  — the closer pair
    pair_hi  — ((task_k, worker_k), d_hi)  — the farther pair

    Returns the upper bound on ε that preserves d_lo < d_hi after noise.
    """
    (ti, wi), d_lo = pair_lo
    (tk, wk), d_hi = pair_hi

    D_lo = d_lo ** 2   # paper notation: D_ij = d_ij^2
    D_hi = d_hi ** 2

    x_diff_lo = abs(tasks[ti][0] - workers[wi][0])
    x_diff_hi = abs(tasks[tk][0] - workers[wk][0])

    denom = x_diff_lo + x_diff_hi

    if denom == 0:
        # Both pairs have the same x-coordinate → case xi = xj, xk = xl
        diff = D_hi - D_lo
        return math.sqrt(diff) if diff > 0 else float("inf")

    return (D_hi - D_lo) / (2 * denom)


def compute_zeta(tasks, workers):
    """
    Compute ζ for one task/worker distribution.

    ζ = min over all consecutive distance pairs of H_{ijkl}
    (Lemma 2 — the tightest bound on noise magnitude that preserves distance order)
    """
    pairs = _all_task_worker_distances(tasks, workers)
    zeta = float("inf")
    for idx in range(len(pairs) - 1):
        h = _h_ijkl(pairs[idx], pairs[idx + 1], tasks, workers)
        if h < zeta:
            zeta = h
    return zeta


# ── Main simulation loop ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Estimate practical ζ (Lemma 2)")
    parser.add_argument(
        "--simulations", type=int, default=10_000,
        help="Number of Monte-Carlo simulations (paper: 10,000)",
    )
    parser.add_argument(
        "--n-tasks", type=int, default=33,
        help="Tasks sampled per simulation (paper: 33)",
    )
    parser.add_argument(
        "--n-workers", type=int, default=67,
        help="Workers sampled per simulation (paper: 67)",
    )
    args = parser.parse_args()

    print(f"Loading Gowalla dataset from {config.DATA_FILE} …")
    all_workers, all_tasks = load_gowalla(
        config.DATA_FILE, config.DATA_CACHE,
        lon_min=config.SF_LON_MIN, lon_max=config.SF_LON_MAX,
        lat_min=config.SF_LAT_MIN, lat_max=config.SF_LAT_MAX,
        ref_lat=config.REF_LAT,    ref_lon=config.REF_LON,
    )
    print(f"  {len(all_tasks):,} tasks, {len(all_workers):,} workers available.")

    n_sims  = args.simulations
    n_tasks  = args.n_tasks
    n_workers = args.n_workers

    zeta_values = []
    for s in range(1, n_sims + 1):
        t_sample = sample(all_tasks,   n_tasks)
        w_sample = sample(all_workers, n_workers)
        z = compute_zeta(t_sample, w_sample)
        zeta_values.append(z)
        if s % max(1, n_sims // 20) == 0:
            print(
                f"  [{s:>{len(str(n_sims))}}/{n_sims}]  "
                f"current min ζ = {min(zeta_values):.6f}",
                flush=True,
            )

    min_zeta = min(zeta_values)
    avg_zeta = np.mean(zeta_values)

    print()
    print(f"Simulations : {n_sims:,}  ({n_tasks} tasks + {n_workers} workers each)")
    print(f"min ζ       : {min_zeta:.6f}   (paper reports 0.0005 for 10,000 sims)")
    print(f"mean ζ      : {avg_zeta:.6f}")


if __name__ == "__main__":
    main()