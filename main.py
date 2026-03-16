#!/usr/bin/env python3
"""
main.py — MPI entry point for PLTA vs LDPP comparison experiments.

Launch with:
    mpirun -n <n_tasks + n_workers + 1> python main.py

Each MPI rank represents one mobile node (task or worker).
Rank 0 acts as the coordinator: it samples positions, broadcasts them,
collects perturbed coordinates, runs the ILP, and writes results.
"""

import os
import sys
import json
import pickle
from time import process_time
from random import sample

import numpy as np
from mpi4py import MPI

import config
from src.data_loader import load_gowalla
from src.helpers import distance_matrix, ilp_min_distance, order_distances, count_order_changes
from src.algorithms import midextrem, create_random_comm_matrix, ldpp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ── Derived parameters ────────────────────────────────────────────────────────
m = config.NUM_TASKS          # number of task-nodes
n = size - m                  # number of worker-nodes (one MPI rank each)
matching_type = (config.TASK_REQUIREMENT, config.WORKER_CAPACITY)


# ── Results accumulator ───────────────────────────────────────────────────────

def _empty_result():
    return {key: [] for key in [
        "n5_cumulative", "n15_cumulative", "n30_cumulative",
        "ldpp_cumulative", "true_cumulative",
        "n5_average",     "n15_average",     "n30_average",
        "ldpp_average",   "true_average",
        "n5_match_changes",  "n15_match_changes",  "n30_match_changes",
        "ldpp_match_changes",
        "n5_sim_changes",    "n15_sim_changes",    "n30_sim_changes",
        "ldpp_sim_changes",
        "n5_time",  "n15_time",  "n30_time",
        "n5_total_time", "n15_total_time", "n30_total_time",
        "ldpp_time",
    ]}


def _append_result(store, sim_result):
    for key, val in sim_result.items():
        store[key].append(val)


def _save(store, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
        for key in store:
            existing.setdefault(key, [])
            existing[key].extend(store[key])
        store = existing
    with open(path, "w") as f:
        json.dump(store, f, indent=2)


# ── Main simulation loop ──────────────────────────────────────────────────────

def main():
    # Load dataset (only rank 0 reads from disk; others receive via broadcast)
    if rank == 0:
        all_workers, all_tasks = load_gowalla(
            config.DATA_FILE, config.DATA_CACHE,
            lon_min=config.SF_LON_MIN, lon_max=config.SF_LON_MAX,
            lat_min=config.SF_LAT_MIN, lat_max=config.SF_LAT_MAX,
            ref_lat=config.REF_LAT,    ref_lon=config.REF_LON,
        )
        print(f"[rank 0] Loaded {len(all_tasks)} tasks, "
              f"{len(all_workers)} workers from Gowalla.")
    else:
        all_tasks = all_workers = None

    all_tasks   = comm.bcast(all_tasks,   root=0)
    all_workers = comm.bcast(all_workers, root=0)

    results_store = _empty_result()
    sim = 0

    while sim < config.NUM_SIMULATIONS:
        sim += 1
        if rank == 0:
            print(f"Simulation {sim}/{config.NUM_SIMULATIONS}", flush=True)

        # ── Sample positions ──────────────────────────────────────────────────
        if rank == 0:
            comm_matrix    = create_random_comm_matrix(size)
            tasks_pos      = np.array(sample(all_tasks,   m))
            workers_pos    = np.array(sample(all_workers, n))
        else:
            comm_matrix = np.empty((size, size), dtype=int)
            tasks_pos   = np.empty((m, 2),       dtype=float)
            workers_pos = np.empty((n, 2),       dtype=float)

        comm_matrix = comm.bcast(comm_matrix, root=0)
        tasks_pos   = comm.bcast(tasks_pos,   root=0)
        workers_pos = comm.bcast(workers_pos, root=0)

        tasks   = [(i, tuple(tasks_pos[i]))   for i in range(m)]
        workers = [(i, tuple(workers_pos[i])) for i in range(n)]

        # ── PLTA: run MidExtrem for each iteration count ──────────────────────
        plta_results = {}
        for iters in config.PLTA_ITERATIONS:
            t0 = process_time() if rank == 0 else None
            offsets = midextrem(iters, config.COORD_RANGES, comm_matrix)
            elapsed_mid = (process_time() - t0) if rank == 0 else None

            # Each rank applies its offset to its own position
            if rank < m:                       # task node
                perturbed_task   = [rank, [tasks_pos[rank][0]   + offsets[0],
                                           tasks_pos[rank][1]   + offsets[1]]]
                perturbed_worker = None
            else:                              # worker node
                wi = rank - m
                perturbed_task   = None
                perturbed_worker = [wi, [workers_pos[wi][0] + offsets[0],
                                         workers_pos[wi][1] + offsets[1]]]

            # Gather perturbed positions at rank 0
            gathered_tasks   = comm.gather(perturbed_task,   root=0)
            gathered_workers = comm.gather(perturbed_worker, root=0)

            if rank == 0:
                noisy_tasks   = [g[1] for g in gathered_tasks   if g is not None]
                noisy_workers = [g[1] for g in gathered_workers if g is not None]

                true_D  = distance_matrix(
                    [t for t in tasks_pos],
                    [w for w in workers_pos],
                )
                noisy_D = distance_matrix(noisy_tasks, noisy_workers)

                t1 = process_time()
                matching, cum, avg = ilp_min_distance(
                    noisy_tasks, noisy_workers, true_D, noisy_D
                )
                elapsed_ilp = process_time() - t1

                plta_results[iters] = {
                    "matching": matching,
                    "cum": cum, "avg": avg,
                    "noisy_tasks":   noisy_tasks,
                    "noisy_workers": noisy_workers,
                    "time_mid": elapsed_mid,
                    "time_total": elapsed_mid + elapsed_ilp,
                }

        # ── True (no-privacy) matching ─────────────────────────────────────────
        if rank == 0:
            true_D = distance_matrix(
                [t for t in tasks_pos],
                [w for w in workers_pos],
            )
            true_matching, true_cum, true_avg = ilp_min_distance(
                tasks_pos.tolist(), workers_pos.tolist(), true_D, true_D
            )

        # ── LDPP baseline ──────────────────────────────────────────────────────
        ldpp_ok = False
        if rank == 0:
            t0 = process_time()
            try:
                ldpp_matching, ldpp_cum, ldpp_avg = ldpp(
                    workers, tasks,
                    epsilon=config.EPSILON,
                    epsilon_max=config.EPSILON_MAX,
                    d_max=config.D_MAX,
                    beta=config.BETA,
                    P_0=config.P_0,
                )
                ldpp_time = process_time() - t0
                ldpp_ok = True
            except Exception as e:
                print(f"  LDPP failed (sim {sim}): {e} — retrying")
                sim -= 1   # do not count this simulation

        ldpp_ok = comm.bcast(ldpp_ok, root=0)
        if not ldpp_ok:
            continue

        # ── Aggregate metrics (rank 0 only) ────────────────────────────────────
        if rank == 0:
            def _match_changes(noisy_m):
                """Count how many (task,worker) assignments changed vs true."""
                diff = np.add(true_matching, noisy_m)
                unique, counts = np.unique(diff, return_counts=True)
                d = dict(zip(unique, counts))
                return float(d.get(1, 0)), float(1 if 1 in d else 0)

            sim_result = {
                "true_cumulative": float(np.mean(true_cum)),
                "true_average":    float(np.mean(true_avg)),
                "ldpp_cumulative": float(np.mean(ldpp_cum)),
                "ldpp_average":    float(np.mean(ldpp_avg)),
                "ldpp_time":       float(np.mean(ldpp_time)),
                "ldpp_match_changes": _match_changes(ldpp_matching)[0],
                "ldpp_sim_changes":   _match_changes(ldpp_matching)[1],
            }
            for iters in config.PLTA_ITERATIONS:
                r = plta_results[iters]
                mc, sc = _match_changes(r["matching"])
                key    = f"n{iters}"
                sim_result[f"{key}_cumulative"]   = float(np.mean(r["cum"]))
                sim_result[f"{key}_average"]      = float(np.mean(r["avg"]))
                sim_result[f"{key}_match_changes"]= mc
                sim_result[f"{key}_sim_changes"]  = sc
                sim_result[f"{key}_time"]         = r["time_mid"]
                sim_result[f"{key}_total_time"]   = r["time_total"]

            _append_result(results_store, sim_result)

            # Flush to disk every 50 simulations
            if sim % 50 == 0:
                _save(results_store, config.OUTPUT_FILE)
                results_store = _empty_result()

    # Final flush
    if rank == 0:
        _save(results_store, config.OUTPUT_FILE)
        print(f"Done. Results written to {config.OUTPUT_FILE}")


if __name__ == "__main__":
    main()
