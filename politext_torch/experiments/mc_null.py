# politext_torch/experiments/mc_null.py
"""Experiment C: behaviour under phi = 0 (no true partisanship).

Expected: MLE biased above 0.5; leave-out and penalized centered on 0.5.

Run: python -m politext_torch.experiments.mc_null
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from politext_torch.estimators import (
    LeaveOutEstimator, MLEEstimator, PenalizedEstimator,
)
from politext_torch.simulate import make_mc_C

N_REP = 100
V = 500
T = 3
N = 600
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
        dgp = make_mc_C(V=V, T=T, N=N, seed=rep)
        for name, est in [
            ("MLE", MLEEstimator(max_iter=30, device=DEVICE)),
            ("LeaveOut", LeaveOutEstimator(device=DEVICE)),
            ("Penalized", PenalizedEstimator(grid_size=10, max_iter=80, device=DEVICE)),
        ]:
            est.fit(dgp["counts"], dgp["party"], dgp["session"])
            pi_hat = est.partisanship_
            for t in range(T):
                rows.append({"rep": rep, "estimator": name,
                             "session": t, "pi_hat": pi_hat[t]})
        print(f"rep {rep+1}/{N_REP}  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results_c.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for est in df["estimator"].unique():
        sub = df[df["estimator"] == est]
        ax.hist(sub["pi_hat"].values, bins=30, alpha=0.4, label=est, density=True)
    ax.axvline(0.5, color="k", lw=1)
    ax.set_xlabel(r"$\hat\pi_t$"); ax.set_ylabel("density")
    ax.set_title(f"Experiment C: null (φ=0), V={V}, N={N}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_c.pdf")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    run()
