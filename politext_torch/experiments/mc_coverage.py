# politext_torch/experiments/mc_coverage.py
"""Experiment B: 95% CI coverage from speaker-level subsampling.

Expected: coverage ≈ 0.95 for leave-out and penalized.

Run: python -m politext_torch.experiments.mc_coverage
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from politext_torch.estimators import LeaveOutEstimator, PenalizedEstimator
from politext_torch.inference import subsample_ci
from politext_torch.simulate import make_mc_B

N_REP = 100
N_SUB = 50          # smaller than paper's 100 for tractability
FRAC = 0.1
OUT = Path(__file__).resolve().parent / "output"
DEVICE = os.environ.get(
    "POLITEXT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)


def run():
    OUT.mkdir(exist_ok=True)
    print(f"device: {DEVICE}")
    rows = []
    for rep in range(N_REP):
        t0 = time.time()
        dgp = make_mc_B(V=100, T=5, N=600, seed=rep)
        true_pi = dgp["true_pi"]
        for name, factory in [
            ("LeaveOut", lambda: LeaveOutEstimator(device=DEVICE)),
            ("Penalized", lambda: PenalizedEstimator(grid_size=8, max_iter=60, device=DEVICE)),
        ]:
            result = subsample_ci(
                factory, dgp["counts"], dgp["party"], dgp["session"],
                n_subsamples=N_SUB, frac=FRAC, seed=rep, transform="log",
            )
            for t in range(len(true_pi)):
                lo, hi = result["ci_lower"][t], result["ci_upper"][t]
                rows.append({
                    "rep": rep, "estimator": name, "session": t,
                    "pi_true": true_pi[t],
                    "estimate": result["estimate"][t],
                    "ci_lower": lo, "ci_upper": hi,
                    "covered": int(lo <= true_pi[t] <= hi)
                                if np.isfinite(lo) and np.isfinite(hi) else 0,
                    "width": (hi - lo) if np.isfinite(lo) and np.isfinite(hi) else np.nan,
                })
        print(f"rep {rep+1}/{N_REP}  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results_b.csv", index=False)

    agg = (df.groupby(["estimator", "session"])
             .agg(coverage=("covered", "mean"),
                  median_width=("width", "median"))
             .reset_index())
    agg.to_csv(OUT / "results_b_agg.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for est in agg["estimator"].unique():
        sub = agg[agg["estimator"] == est]
        axes[0].plot(sub["session"], sub["coverage"], marker="o", label=est)
        axes[1].plot(sub["session"], sub["median_width"], marker="o", label=est)
    axes[0].axhline(0.95, color="k", lw=0.5, ls="--")
    axes[0].set_xlabel("session"); axes[0].set_ylabel("95% CI coverage")
    axes[0].set_ylim(0.5, 1.05); axes[0].legend()
    axes[1].set_xlabel("session"); axes[1].set_ylabel("Median CI width"); axes[1].legend()
    fig.suptitle("Experiment B: 95% CI coverage via speaker subsampling")
    fig.tight_layout()
    fig.savefig(OUT / "fig_b.pdf")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    run()
