#!/usr/bin/env python3
"""
output/aggregate_and_plot.py — Load raw results JSON and produce figures.

Usage:
    python output/aggregate_and_plot.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

RESULTS_FILE = config.OUTPUT_FILE
FIGURES_DIR  = config.FIGURES_DIR
os.makedirs(FIGURES_DIR, exist_ok=True)

with open(RESULTS_FILE) as f:
    data = json.load(f)

means = {k: np.mean(v) for k, v in data.items() if v}

# ── Helper ────────────────────────────────────────────────────────────────────

def bar_chart(title, ylabel, values_dict, filename):
    labels = list(values_dict.keys())
    vals   = [values_dict[k] for k in labels]
    plt.figure()
    plt.bar(labels, vals, color=["blue", "orange", "green", "red", "black"])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename))
    plt.close()

# ── Cumulative distance ───────────────────────────────────────────────────────
bar_chart(
    "Cumulative Travel Distance",
    "Distance (m)",
    {
        "PLTA-5":   means.get("n5_cumulative",   0),
        "PLTA-15":  means.get("n15_cumulative",  0),
        "PLTA-30":  means.get("n30_cumulative",  0),
        "LDPP":     means.get("ldpp_cumulative", 0),
        "No priv.": means.get("true_cumulative", 0),
    },
    "cumulative_distance.png",
)

# ── Average distance ──────────────────────────────────────────────────────────
bar_chart(
    "Average Travel Distance",
    "Distance (m)",
    {
        "PLTA-5":   means.get("n5_average",   0),
        "PLTA-15":  means.get("n15_average",  0),
        "PLTA-30":  means.get("n30_average",  0),
        "LDPP":     means.get("ldpp_average", 0),
        "No priv.": means.get("true_average", 0),
    },
    "average_distance.png",
)

# ── Matching changes ──────────────────────────────────────────────────────────
bar_chart(
    "Average Number of Matching Changes",
    "# assignments changed",
    {
        "PLTA-5":  means.get("n5_match_changes",   0),
        "PLTA-15": means.get("n15_match_changes",  0),
        "PLTA-30": means.get("n30_match_changes",  0),
        "LDPP":    means.get("ldpp_match_changes", 0),
    },
    "matching_changes.png",
)

# ── Execution time ────────────────────────────────────────────────────────────
bar_chart(
    "Execution Time (perturbation only)",
    "Time (s)",
    {
        "PLTA-5":  means.get("n5_time",   0),
        "PLTA-15": means.get("n15_time",  0),
        "PLTA-30": means.get("n30_time",  0),
        "LDPP":    means.get("ldpp_time", 0),
    },
    "execution_time.png",
)

print(f"Figures saved to {FIGURES_DIR}/")
