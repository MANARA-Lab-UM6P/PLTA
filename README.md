# PLTA — Privacy-preserving Location-based Task Allocation

This repository contains the full implementation and evaluation code for **PLTA**, a distributed, privacy-preserving task allocation algorithm for Mobile Crowdsourcing Systems (MCS).

PLTA is compared against **LDPP** (Location Differential Privacy Protocol), a Voronoi-based baseline that applies the Location Coding Obfuscation (LCO) mechanism with differential privacy.

---

## Overview

In MCS platforms (e.g., spatial task apps), workers share their locations with a central server to be matched to nearby tasks. This raises serious privacy concerns.

**PLTA** protects worker and task locations through a distributed perturbation protocol called **MidExtrem**:
- Each node (task or worker) is assigned to one MPI process.
- Over *N* communication rounds, processes exchange coordinate estimates with their neighbours, identify the extremal pair (max absolute difference), and update to the midpoint.
- The resulting offset is added to each node's true coordinate before any matching is performed.

**LDPP** (baseline):
- Constructs a Voronoi diagram partitioned by task locations.
- Each worker determines its Voronoi region and obfuscates its location using LCO + differential privacy (ε-LDP).
- An ILP matches workers to tasks maximising acceptance probability.

Both approaches are evaluated using an **ILP-based optimal matching** on the (possibly perturbed) location data.

---

## Repository structure

```
PLTA/
├── main.py                    # MPI entry point — runs the full experiment
├── config.py                  # All experiment parameters in one place
├── requirements.txt
├── run.sh                     # SLURM job script
│
├── src/
│   ├── algorithms.py          # MidExtrem (PLTA) and LDPP implementations
│   ├── helpers.py             # Distance utilities, ILP solvers (OR-Tools)
│   └── data_loader.py         # Gowalla dataset loading & coordinate conversion
│
├── data/
│   ├── data_generation.py     # Placeholder — dataset generation (TODO)
│   └── loc-gowalla_totalCheckins.txt   # ← provide this file (see below)
│
└── output/
    ├── aggregate_and_plot.py  # Load results JSON and generate figures
    └── figures/               # Generated plots (created at runtime)
```

---

## Dataset

The experiments use the **Gowalla** location check-in dataset, filtered to the San Francisco metropolitan area.

1. Download the raw file from [SNAP](https://snap.stanford.edu/data/loc-Gowalla.html):
   `loc-gowalla_totalCheckins.txt.gz`
2. Decompress and place it at `data/loc-gowalla_totalCheckins.txt`.

A pickle cache (`data/gowalla_cache.pkl`) is created automatically on first run.

> **Note:** `data_generation.py` is a placeholder for a future script that will automate this step.

---

## Installation

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Required packages:

| Package | Purpose |
|---------|---------|
| `mpi4py` | Distributed MPI communication (PLTA) |
| `numpy` | Numerical operations |
| `scipy` | Voronoi diagram (LDPP) |
| `shapely` | Point-in-polygon test (LDPP) |
| `ortools` | ILP matching solver |
| `pandas` | Dataset loading |
| `matplotlib` | Result plotting |
| `pyproj` | Coordinate projection |

---

## Running the experiments

### Local (small test)

```bash
# 21 MPI ranks = 10 workers + 10 tasks + 1 coordinator
mpirun -n 21 python main.py
```

Adjust `NUM_TASKS` in `config.py` to keep the total ranks ≤ your CPU count.

### HPC cluster (SLURM)

Edit `run.sh` to match your cluster's module names and partition, then:

```bash
sbatch run.sh
```

The default configuration uses **201 MPI ranks** (100 tasks + 100 workers + 1 coordinator) and **1 300 Monte-Carlo simulations**.

---

## Generating figures

Once the experiment has finished:

```bash
python output/aggregate_and_plot.py
```

Figures are saved to `output/figures/`:

| File | Metric |
|------|--------|
| `cumulative_distance.png` | Total travel distance per algorithm |
| `average_distance.png` | Average travel distance per worker |
| `matching_changes.png` | Number of task–worker assignments that changed |
| `execution_time.png` | Runtime of the perturbation step |

---

## Configuration

All parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_SIMULATIONS` | 1300 | Monte-Carlo repetitions |
| `NUM_TASKS` | 100 | Tasks per simulation |
| `PLTA_ITERATIONS` | [5, 15, 30] | MidExtrem iteration counts |
| `COORD_RANGES` | ±1 000 000 m | Range for random initial coordinates |
| `EPSILON` | 1.0 | Differential privacy budget (LDPP) |
| `D_MAX` | 10 000 m | Maximum acceptance distance (LDPP) |

---

## Citation

If you use this code in your research, please cite the associated paper (link to be added).

---

## License

MIT
