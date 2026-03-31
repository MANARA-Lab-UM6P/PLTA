# PLTA — Privacy-preserving Location-based Task Allocation

> **This repository is the official implementation of:**
>
> **"PLTA: Private Location Task Allocation using multidimensional approximate agreement"**
> *2024 IEEE Conference on Communications and Network Security (CNS)*
> DOI: [10.1109/CNS62487.2024.10735598](https://doi.org/10.1109/CNS62487.2024.10735598)

---

## Overview

PLTA is a distributed, privacy-preserving task allocation algorithm for Mobile Crowdsensing (MCS).
Workers and tasks run the **MidExtremes** approximate-agreement protocol to collectively agree on
a noise offset, which each node adds to its own x-coordinate before submitting its location to the
server. The server then solves an ILP matching on the perturbed locations.

PLTA is compared against **LDPP**, a Voronoi + Local Differential Privacy baseline from [Zhang et al., 2024].

---

## Repository structure

```
PLTA/
├── main.py                  # MPI entry point — runs the full experiment
├── config.py                # All experiment parameters in one place
├── requirements.txt
├── run.sh                   # SLURM job script (HPC)
│
├── src/
│   ├── algorithms.py        # MidExtremes (PLTA) and LDPP implementations
│   ├── helpers.py           # Distance utilities and ILP solver (OR-Tools)
│   └── data_loader.py       # Gowalla dataset loading & coordinate conversion
│
├── data/
│   ├── data_generation.py   # Downloads and extracts the Gowalla dataset
│   ├── compute_zeta.py      # Estimates the practical noise threshold ζ (Section VI-B)
│   └── loc-gowalla_totalCheckins.txt   # created by data_generation.py
│
└── output/
    ├── aggregate_and_plot.py  # Loads results JSON and generates figures
    └── figures/               # Generated plots (created at runtime)
```

---

## Quick Start

### 1. Set up the parallel computing environment

The experiment requires MPI. Before anything else, load the MPI and Python modules
for your system.

**On the Toubkal HPC cluster**, first check which modules are available:

```bash
module avail OpenMPI
module avail mpi4py
```

Then load the `mpi4py` module that bundles both OpenMPI and Python (recommended — it
loads all required dependencies in one step):

```bash
module load mpi4py/4.0.1-gompi-2024a
```

> Substitute the version with whichever is listed on your cluster.
> This automatically loads `OpenMPI/5.0.3-GCC-13.3.0` and `Python/3.12.3`.

**If you use SLURM as your workload manager**, you must run the experiment inside a
job allocation that reserves enough CPU slots. The largest configuration needs
180 MPI ranks (80 tasks + 100 workers). Request an interactive session before
running `mpirun`:

```bash
# Reserve 180 cores on one or more nodes, then get an interactive shell
srun --ntasks=180 --cpus-per-task=1 --pty bash

# Or submit all four configurations as a batch job
sbatch run.sh
```

> Without a SLURM allocation, `mpirun` will refuse to start more processes than
> the number of physical cores on the login node. Use `run.sh` as a template for
> the batch script.

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Download the dataset

```bash
python data/data_generation.py
```

This downloads `loc-gowalla_totalCheckins.txt.gz` (~100 MB) from the
[Stanford SNAP](https://snap.stanford.edu/data/loc-Gowalla.html) repository,
extracts it to `data/loc-gowalla_totalCheckins.txt`, and deletes the archive.
Re-running the script is safe — it skips the download if the file already exists.

### 5. (Optional) Reproduce the practical ζ threshold (Section VI-B)

The paper estimates the maximum safe noise magnitude ζ by running 10,000 simulations
sampling 33 tasks + 67 workers each from the Gowalla dataset, and taking the minimum:

```bash
python data/compute_zeta.py --simulations 10000
# Expected output:  min ζ ≈ 0.0005
```

This is informational only — `config.py` already encodes the result and the
main experiment does not call this script.

### 6. Run the experiments

The paper evaluates PLTA with **workers fixed at 100** and the **number of tasks
varying over {20, 40, 60, 80}** (1,000 simulations per configuration, 4,000 total).

In the code, `NUM_TASKS` is the number of task-nodes and the number of worker-nodes
equals `MPI_size − NUM_TASKS`. Rank 0 acts as both coordinator and task-node 0.

Run each of the four configurations:

```bash
# n = 20 tasks, 100 workers  →  20 + 100 = 120 MPI ranks
NUM_TASKS=20  mpirun -n 120 python main.py

# n = 40 tasks, 100 workers  →  140 ranks
NUM_TASKS=40  mpirun -n 140 python main.py

# n = 60 tasks, 100 workers  →  160 ranks
NUM_TASKS=60  mpirun -n 160 python main.py

# n = 80 tasks, 100 workers  →  180 ranks
NUM_TASKS=80  mpirun -n 180 python main.py
```

> **How `NUM_TASKS` is passed:** the value is read from `config.py`.
> Edit `config.py` and set `NUM_TASKS` to the desired value before each run,
> then adjust the `-n` argument to `mpirun` accordingly: `-n NUM_TASKS + 100`.

Set `NUM_SIMULATIONS = 1000` in `config.py` for each run (paper default).

Results from all runs are appended to the same `output/raw_results.json` file.

### 7. Generate figures

```bash
python output/aggregate_and_plot.py
```

Figures are saved to `output/figures/`.

---

## Pre-computed results (no cluster required)

If you do not have access to an MPI cluster, the `data/precomputed/` folder
(coming soon) will contain ready-to-use result files covering a range of
(**n tasks**, **m workers**) configurations — not limited to the paper's settings.

Each JSON file includes:
- The **worker and task spatial distributions** used in each simulation (sampled
  from the Gowalla dataset), so results are fully reproducible and can serve as
  input to your own implementation.
- The **per-simulation outputs** of all five algorithms: **PLTA (T = 5)**,
  **PLTA (T = 15)**, **PLTA (T = 30)**, **LDPP**, and **No-privacy** (optimal ILP
  lower bound).

You will be able to load any of these files directly into
`output/aggregate_and_plot.py` to reproduce figures for the corresponding
(n, m) configuration without running the experiment yourself.

---

## Dataset

The Gowalla check-in dataset is filtered to the **San Francisco area**
(lon ∈ [−122.52, −122.36], lat ∈ [37.70, 37.84]).

After preprocessing (`data_loader.py`):
- **82,702 workers** — each user's most recent check-in
- **1,173,991 tasks** — all earlier check-in locations

A pickle cache (`data/gowalla_cache.pkl`) is created automatically on the first
run of `main.py` and reused on subsequent runs.

---

## HPC cluster (SLURM)

Edit `run.sh` to match your cluster's module names and partition, set
`NUM_TASKS` in `config.py`, and submit:

```bash
sbatch run.sh
```

The default `run.sh` uses 201 ranks (100 tasks + 100 workers) as a starting point;
adjust `--ntasks` to match the configuration you are running.

---

## Configuration (`config.py`)

| Parameter | Paper value | Description |
|-----------|-------------|-------------|
| `NUM_SIMULATIONS` | 1000 | Monte-Carlo repetitions per configuration |
| `NUM_TASKS` | 20 / 40 / 60 / 80 | Number of task-nodes (= n in the paper) |
| `PLTA_ITERATIONS` | [5, 15, 30] | MidExtremes iteration counts (= T) |
| `COORD_RANGES` | ±1 000 000 m | Initial value range for the agreement protocol (Δ = 2 000 000) |
| `EPSILON` | 1.0 | Differential privacy budget (LDPP baseline) |
| `D_MAX` | 10 000 m | Maximum acceptance distance (LDPP baseline) |

> **Note on notation:** the paper uses *m* for workers (fixed at 100) and *n* for
> tasks (varied). The code uses `NUM_TASKS` for the task count; the worker count
> is derived as `MPI_size − NUM_TASKS`.

---

## Metrics

| Metric | Description |
|--------|-------------|
| Average distance traveled | Mean worker travel distance after ILP matching |
| Number of matching changes | Pairs that differ from the no-privacy optimal matching |
| Running time | Wall-clock time of the perturbation (agreement) step |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `mpi4py` | Distributed MPI communication (PLTA) |
| `numpy` | Numerical operations |
| `scipy` | Voronoi diagram (LDPP) |
| `shapely` | Point-in-polygon test (LDPP) |
| `ortools` | ILP matching solver |
| `pandas` | Dataset loading |
| `matplotlib` | Result plotting |

---

## Citation

```bibtex
@inproceedings{plta2024,
  title     = {PLTA: Private Location Task Allocation using multidimensional approximate agreement},
  booktitle = {2024 IEEE Conference on Communications and Network Security (CNS)},
  year      = {2024},
  doi       = {10.1109/CNS62487.2024.10735598},
  url       = {https://doi.org/10.1109/CNS62487.2024.10735598}
}
```

---

## License

MIT
