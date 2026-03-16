# =============================================================================
# src/algorithms.py — PLTA and LDPP algorithm implementations
# =============================================================================

import math
import random
import numpy as np
from mpi4py import MPI
from shapely.geometry import Point, Polygon

from src.helpers import (
    compute_distance, distance_matrix,
    ilp_min_distance, ilp_max_probability,
    _fmt, get_bounds,
)


# ── PLTA: Privacy-preserving Location-based Task Allocation ──────────────────

def _find_extremal_pair(values):
    """Return the pair (a, b) with the maximum absolute difference."""
    max_diff = -1
    pair = (None, None)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            diff = abs(values[i] - values[j])
            if diff > max_diff:
                max_diff = diff
                pair = (values[i], values[j])
    return pair, max_diff


def midextrem(num_iterations, coord_ranges, comm_matrix):
    """
    MidExtrem distributed perturbation algorithm (PLTA core).

    Each MPI process holds one node (task or worker).  Over
    *num_iterations* rounds the process exchanges its coordinate
    estimate with its neighbours (defined by *comm_matrix*), finds
    the extremal pair among received values, and updates its estimate
    to their midpoint.  The resulting value is the noise offset added
    to the true coordinate.

    Parameters
    ----------
    num_iterations : int
    coord_ranges   : list of two (min, max) tuples — one per axis
    comm_matrix    : 2-D int array — comm_matrix[i][j] == 1 means i sends to j

    Returns
    -------
    values : [x_offset, y_offset]  (one float per axis)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    values = [
        np.random.uniform(*coord_ranges[0]),
        np.random.uniform(*coord_ranges[1]),
    ]

    for _ in range(num_iterations):
        received = []

        for j in range(size):
            if comm_matrix[rank][j] == 1:
                comm.send(values, dest=j)

        for j in range(size):
            if comm_matrix[j][rank] == 1:
                received.append(comm.recv(source=j))

        for axis in range(2):
            extremal, _ = _find_extremal_pair([v[axis] for v in received])
            values[axis] = float(np.mean(extremal))

    return values


def create_random_comm_matrix(size):
    """Return a random symmetric binary communication matrix."""
    M = np.zeros((size, size), dtype=int)
    for i in range(size):
        conns = np.random.choice(size, np.random.randint(1, size), replace=False)
        conns = conns[conns != i]
        M[i, conns] = 1
        M[conns, i] = 1
    return M


# ── LDPP: Location Differential Privacy Protocol (Voronoi + LCO) ─────────────

def _is_in_box(point, bbox):
    x, y = point
    xmin, xmax, ymin, ymax = bbox
    return xmin <= x <= xmax and ymin <= y <= ymax


def _all_in_box(points, bbox):
    return all(_is_in_box(p, bbox) for p in points)


def _line_intersect(box, start, end):
    """Intersect line segment [start, end] with the bounding box."""
    bounds = [box[0], box[2], box[1], box[3]]
    directions = [np.array([1, 0]), np.array([1, 0]),
                  np.array([0, 1]), np.array([0, 1])]
    hits = []
    for d, bound in zip(directions, bounds):
        denom = (end - start) @ d
        if denom == 0:
            continue
        t = _fmt((bound - (start @ d)) / denom, 5)
        pt = start + ((bound - (start @ d)) / denom) * (end - start)
        p0, p1 = _fmt(pt[0]), _fmt(pt[1])
        if box[0] <= p0 <= box[2] and box[1] <= p1 <= box[3] and 0 <= t <= 1:
            hits.append(np.array([p0, p1]))
    return hits


def _order_vertices(vertex_indices, all_vertices):
    verts = [all_vertices[vi] for vi in vertex_indices]
    cx = sum(v[0] for v in verts) / len(verts)
    cy = sum(v[1] for v in verts) / len(verts)
    return sorted(vertex_indices,
                  key=lambda vi: math.atan2(all_vertices[vi][1] - cy,
                                            all_vertices[vi][0] - cx))


def _clip_voronoi(tasks, box):
    """
    Build a bounded Voronoi diagram for *tasks* clipped to *box*.

    Returns
    -------
    regions  : dict  task_id → ordered list of vertex indices
    vertices : dict  index   → np.array([x, y])
    """
    from scipy.spatial import Voronoi

    bounds = [box[0], box[2], box[1], box[3]]   # xmin xmax ymin ymax
    points = np.array([coord for _, coord in tasks])
    task_ids = [t_id for t_id, _ in tasks]

    vor = Voronoi(points)
    center = vor.points.mean(axis=0)

    # Keep only Voronoi vertices inside the box
    regions = {i: set() for i in range(len(task_ids))}
    vertices = {
        i: vor.vertices[i]
        for i in range(len(vor.vertices))
        if _is_in_box(vor.vertices[i], bounds)
    }
    nv = len(vor.vertices)   # next free vertex index

    for pidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        intersections = []

        if np.any(simplex < 0):
            # Infinite ridge — project to far point then intersect box
            i = simplex[simplex >= 0][0]
            t_vec = vor.points[pidx[1]] - vor.points[pidx[0]]
            t_vec /= np.linalg.norm(t_vec)
            n_vec = np.array([-t_vec[1], t_vec[0]])
            mid   = vor.points[pidx].mean(axis=0)
            direction = np.sign(np.dot(mid - center, n_vec)) * n_vec
            far = vor.vertices[i] + direction * max(box[2], box[3]) * 10
            intersections += _line_intersect(box, vor.vertices[i], far)

        elif not _all_in_box(
                (vor.vertices[simplex[0]], vor.vertices[simplex[1]]), bounds):
            intersections += _line_intersect(
                box, vor.vertices[simplex[0]], vor.vertices[simplex[1]])

        for pt in intersections:
            vertices[nv] = np.array([_fmt(pt[0]), _fmt(pt[1])])
            for rp in pidx:
                regions[rp].add(nv)
            nv += 1

        for rp in pidx:
            regions[rp].update(
                e for e in simplex.tolist()
                if e >= 0 and _is_in_box(vor.vertices[e], bounds)
            )

    # Add box corners if missing
    unused = set()
    for xi in range(2):
        for yi in range(2, 4):
            corner = np.array([bounds[xi], bounds[yi]])
            if all(not np.array_equal(corner, v) for v in vertices.values()):
                vertices[nv] = np.array([_fmt(corner[0]), _fmt(corner[1])])
                useless = True
                for rid in regions:
                    rx = [vertices[vi][0] for vi in regions[rid]]
                    ry = [vertices[vi][1] for vi in regions[rid]]
                    if corner[0] in rx and corner[1] in ry:
                        regions[rid].add(nv)
                        useless = False
                if useless:
                    unused.add(nv)
                nv += 1

    if 1 <= len(unused) <= 2:
        xb = [vertices[vi][0] for vi in unused]
        yb = [vertices[vi][1] for vi in unused]
        for rid in regions:
            rx = [vertices[vi][0] for vi in regions[rid]]
            ry = [vertices[vi][1] for vi in regions[rid]]
            if all(x in rx for x in xb) or all(y in ry for y in yb):
                regions[rid].update(unused)

    # Order vertices counter-clockwise for each region
    for rid in list(regions):
        regions[rid] = _order_vertices(regions[rid], vertices)

    return regions, vertices


def _build_voronoi_map(tasks, workers):
    """Compute the bounded Voronoi diagram and return (vertices, regions, box)."""
    all_pts = [t[1] for t in tasks] + [w[1] for w in workers]
    bbox = get_bounds(all_pts)
    pad_x = abs(_fmt((bbox[2] - bbox[0]) / 8))
    pad_y = abs(_fmt((bbox[3] - bbox[1]) / 8))
    bbox[0] -= pad_x;  bbox[1] -= pad_y
    bbox[2] += pad_x;  bbox[3] += pad_y

    tasks_fmt = [
        [t[0], (_fmt(t[1][0]), _fmt(t[1][1]))]
        for t in tasks
    ]
    regions, vertices = _clip_voronoi(tasks_fmt, bbox)
    return vertices, regions, bbox


def _lco(vertices, worker_loc, epsilon, m=3, s=3):
    """
    Location Coding Obfuscation (LCO) with differential privacy.

    Parameters
    ----------
    vertices    : list of (x, y) polygon corners of the worker's Voronoi region
    worker_loc  : (x, y) true location of the worker
    epsilon     : privacy budget
    m, s        : bits for x and y encoding respectively

    Returns
    -------
    (x_obf, y_obf) : obfuscated location
    codeword       : perturbed bit-string
    """
    min_x = min(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_x = max(v[0] for v in vertices)
    max_y = max(v[1] for v in vertices)

    dx = (max_x - min_x) / (2 ** m)
    dy = (max_y - min_y) / (2 ** s)

    xw, yw = worker_loc
    rel_x = int((xw - min_x) / dx) if dx > 0 else 0
    rel_y = int((yw - min_y) / dy) if dy > 0 else 0

    Ux = bin(rel_x)[2:].zfill(m)
    Uy = bin(rel_y)[2:].zfill(s)
    U  = Ux + Uy

    Q = m + s
    P = max(len(U) // Q, 1)
    U_prime = ""
    for q in range(Q):
        chunk = U[q * P: (q + 1) * P]
        f = 1 / (np.exp(epsilon / Q) + 2 ** P - 1)
        if np.random.uniform(0, 1) < f:
            chunk = "".join("1" if b == "0" else "0" for b in chunk)
        U_prime += chunk

    xi = int(U_prime[:m], 2)
    yi = int(U_prime[m:], 2)
    return (xi * dx + min_x, yi * dy + min_y), U_prime


def ldpp(workers, tasks, epsilon, epsilon_max=7.0, d_max=1e4, beta=1.0, P_0=1.0):
    """
    LDPP: Location Differential Privacy Protocol.

    1. Build a Voronoi partition of the map based on task locations.
    2. Each worker finds its Voronoi region (closest task) and applies LCO.
    3. An ILP assigns workers to tasks maximising acceptance probability.

    Returns
    -------
    matching, cumulative_distance, average_distance
    Raises ValueError if Voronoi construction fails.
    """
    try:
        all_vertices, all_regions, box = _build_voronoi_map(tasks, workers)
    except Exception as exc:
        raise ValueError(f"Voronoi construction failed: {exc}") from exc

    task_ids = [t[0] for t in tasks]
    worker_regions  = {}
    worker_obfu_loc = {}

    for wid, wloc in workers:
        wloc_fmt = (_fmt(wloc[0]), _fmt(wloc[1]))
        for tid in task_ids:
            verts = [
                (_fmt(all_vertices[vi][0]), _fmt(all_vertices[vi][1]))
                for vi in all_regions.get(tid, [])
            ]
            try:
                inside = Polygon(verts).contains(Point(wloc_fmt))
            except Exception:
                raise ValueError("Polygon construction failed — retrying simulation.")
            if inside:
                obf_loc, _ = _lco(verts, wloc, epsilon)
                worker_regions[wid]  = tid
                worker_obfu_loc[wid] = obf_loc

    tasks_pos   = [t[1] for t in tasks]
    workers_pos = [w[1] for w in workers]
    for wid, loc in worker_obfu_loc.items():
        workers_pos[wid] = loc

    true_D = distance_matrix(tasks_pos, [w[1] for w in workers])
    D      = distance_matrix(tasks_pos, workers_pos)
    n, m   = len(tasks), len(workers)

    def _accept_prob(d):
        if 0 <= d <= d_max:
            return (P_0 - beta * d / d_max) * (epsilon / epsilon_max)
        return 0.0

    prob = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if j in worker_regions and worker_regions[j] == task_ids[i]:
                prob[i][j] = _accept_prob(
                    compute_distance(tasks_pos[i], [w[1] for w in workers][j])
                )

    return ilp_max_probability(tasks_pos, workers_pos, true_D, prob)
