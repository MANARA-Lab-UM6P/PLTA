# =============================================================================
# src/helpers.py — Distance utilities and ILP matching solvers
# =============================================================================

import math
import numpy as np
from ortools.linear_solver import pywraplp


# ── Geometry ──────────────────────────────────────────────────────────────────

def compute_distance(node1, node2):
    """Euclidean distance between two 2-D points."""
    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


def distance_matrix(tasks, workers):
    """Return an n×m matrix of Euclidean distances."""
    n, m = len(tasks), len(workers)
    D = [[float("inf")] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if tasks[i] is not None and workers[j] is not None:
                D[i][j] = compute_distance(tasks[i], workers[j])
    return D


def order_distances(tasks, workers):
    """Return task–worker pairs sorted by Euclidean distance (ascending)."""
    pairs = []
    for i, t in enumerate(tasks):
        for j, w in enumerate(workers):
            pairs.append([(i, j), compute_distance(t, w)])
    pairs.sort(key=lambda x: x[1])
    return pairs


def count_order_changes(original, noisy):
    """Count how many (task, worker) pairs changed rank after perturbation."""
    orig_pairs  = [p for p, _ in original]
    noisy_pairs = [p for p, _ in noisy]
    return sum(1 for a, b in zip(orig_pairs, noisy_pairs) if a != b)


def get_bounds(positions):
    """Return [min_x, min_y, max_x, max_y] for a list of (x, y) positions."""
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    return [_fmt(min(xs)), _fmt(min(ys)), _fmt(max(xs)), _fmt(max(ys))]


def _fmt(num, precision=2):
    return float(f"{num:.{precision}f}")


# ── ILP solvers ───────────────────────────────────────────────────────────────

def ilp_min_distance(tasks, workers, true_D, noisy_D,
                     task_req=1, worker_cap=1):
    """
    Minimise total travel distance subject to bipartite matching constraints.

    Uses *noisy_D* to decide the assignment but reports true travel distances
    from *true_D*.

    Returns
    -------
    matching : n×m binary matrix
    cumulative_distance : float
    average_distance : float
    """
    n, m = len(tasks), len(workers)
    if n * task_req > m * worker_cap:
        raise ValueError("Not enough workers to cover all tasks.")

    solver = pywraplp.Solver.CreateSolver("SCIP")
    x = {(i, j): solver.IntVar(0, 1, f"x[{i},{j}]")
         for i in range(n) for j in range(m)}

    # Every task must be covered at least once
    solver.Add(solver.Sum(x[i, j] for i in range(n) for j in range(m))
               >= n * task_req)
    for i in range(n):
        solver.Add(solver.Sum(x[i, j] for j in range(m)) <= task_req)
    for j in range(m):
        solver.Add(solver.Sum(x[i, j] for i in range(n)) <= worker_cap)

    solver.Minimize(
        solver.Sum(noisy_D[i][j] * x[i, j]
                   for i in range(n) for j in range(m))
    )

    status = solver.Solve()
    matching = [[0.0] * m for _ in range(n)]
    travel   = [[0.0] * m for _ in range(n)]
    if status == pywraplp.Solver.OPTIMAL:
        for i in range(n):
            for j in range(m):
                v = x[i, j].solution_value()
                matching[i][j] = v
                travel[i][j]   = true_D[i][j] * v

    cum  = float(np.sum(travel))
    avg  = float(np.mean(np.sum(travel, axis=0)))
    return matching, cum, avg


def ilp_max_probability(tasks, workers, true_D, prob_matrix,
                        task_req=1, worker_cap=1):
    """
    Maximise total acceptance probability subject to bipartite matching.
    Used by the LDPP baseline.

    Returns
    -------
    matching : n×m binary matrix
    cumulative_distance : float   (using true distances)
    average_distance : float
    """
    n, m = len(tasks), len(workers)
    if n * task_req > m * worker_cap:
        raise ValueError("Not enough workers to cover all tasks.")

    solver = pywraplp.Solver.CreateSolver("SCIP")
    x = {(i, j): solver.IntVar(0, 1, f"x[{i},{j}]")
         for i in range(n) for j in range(m)}

    solver.Add(solver.Sum(x[i, j] for i in range(n) for j in range(m))
               >= n * task_req)
    for i in range(n):
        solver.Add(solver.Sum(x[i, j] for j in range(m)) <= task_req)
    for j in range(m):
        solver.Add(solver.Sum(x[i, j] for i in range(n)) <= worker_cap)

    solver.Maximize(
        solver.Sum(prob_matrix[i][j] * x[i, j]
                   for i in range(n) for j in range(m))
    )

    status = solver.Solve()
    matching = [[0.0] * m for _ in range(n)]
    travel   = [[0.0] * m for _ in range(n)]
    if status == pywraplp.Solver.OPTIMAL:
        for i in range(n):
            for j in range(m):
                v = x[i, j].solution_value()
                matching[i][j] = v
                travel[i][j]   = true_D[i][j] * v

    cum = float(np.sum(travel))
    avg = float(np.mean(np.sum(travel, axis=0)))
    return matching, cum, avg
