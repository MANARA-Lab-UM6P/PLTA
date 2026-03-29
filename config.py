# =============================================================================
# config.py — Central configuration for PLTA experiments
# =============================================================================

# ── Dataset ──────────────────────────────────────────────────────────────────
# Raw Gowalla check-in file (tab-separated)
DATA_FILE = "data/loc-gowalla_totalCheckins.txt"
# Preprocessed cache (pickle) – created automatically on first run
DATA_CACHE = "data/gowalla_cache.pkl"

# San Francisco bounding box used to filter check-ins
SF_LON_MIN, SF_LON_MAX = -122.52, -122.36
SF_LAT_MIN, SF_LAT_MAX = 37.70, 37.84
# Flat-earth reference point (central SF)
REF_LAT, REF_LON = 37.7749, -122.4194

# ── Simulation ────────────────────────────────────────────────────────────────
NUM_SIMULATIONS = 1000   # Monte-Carlo repetitions per configuration (paper: 1,000)
# Number of task-nodes (= n in the paper).  Paper evaluates n ∈ {20, 40, 60, 80}.
# Workers (m = 100 in the paper) = MPI_size − NUM_TASKS.
# Set mpirun -n (NUM_TASKS + 100) accordingly.
NUM_TASKS      = 20      # start with n = 20; change to 40 / 60 / 80 for full sweep

# PLTA iteration counts to benchmark
PLTA_ITERATIONS = [5, 15, 30]

# Coordinate ranges for MidExtrem perturbation
COORD_RANGES = [(-1_000_000, 1_000_000),   # x-axis
                (-1_000_000, 1_000_000)]    # y-axis

# ── LDPP (Voronoi + LCO) ─────────────────────────────────────────────────────
EPSILON         = 1.0   # differential privacy budget
EPSILON_MAX     = 7.0
D_MAX           = 1e4   # maximum acceptance distance (metres)
BETA            = 1.0
P_0             = 1.0
LCO_M_BITS      = 3     # horizontal encoding bits
LCO_S_BITS      = 3     # vertical   encoding bits

# ── ILP ───────────────────────────────────────────────────────────────────────
TASK_REQUIREMENT  = 1   # each task needs exactly 1 worker
WORKER_CAPACITY   = 1   # each worker covers at most 1 task

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_FILE    = "output/raw_results.json"
FIGURES_DIR    = "output/figures"
