"""Experiment A: bias and RMSE of π̂ as vocabulary size V grows.

Expected: MLE biased upward in V; leave-out and penalized near truth.

Run: python -m politext_torch.experiments.mc_bias_rmse
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
from politext_torch.simulate import make_mc_A

V_GRID = [50, 200, 500, 1000]
N_REP = 50
N = 800
T = 5
OUT = Path(__file__).resolve().parent / "output"
DEVICE = os.environ.get(
    "POLITEXT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)


def run():
    OUT.mkdir(exist_ok=True)
    print(f"device: {DEVICE}")
    rows = []
    for V in V_GRID:
        print(f"=== V = {V} ===")
        for rep in range(N_REP):
            t0 = time.time()
            dgp = make_mc_A(V=V, T=T, N=N, seed=rep)
            true_pi = dgp["true_pi"]
            for name, est in [
                ("MLE", MLEEstimator(max_iter=30, device=DEVICE)),
                ("LeaveOut", LeaveOutEstimator(device=DEVICE)),
                ("Penalized", PenalizedEstimator(grid_size=15, max_iter=80, device=DEVICE)),
            ]:
                est.fit(dgp["counts"], dgp["party"], dgp["session"])
                pi_hat = est.partisanship_
                for t in range(T):
                    rows.append({
                        "V": V, "rep": rep, "estimator": name, "session": t,
                        "pi_hat": pi_hat[t], "pi_true": true_pi[t],
                        "err": pi_hat[t] - true_pi[t],
                    })
            print(f"  rep {rep+1}/{N_REP}  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results_a.csv", index=False)

    agg = (df.groupby(["V", "estimator"])
             .agg(bias=("err", "mean"),
                  rmse=("err", lambda x: float(np.sqrt(np.mean(x**2)))))
             .reset_index())
    agg.to_csv(OUT / "results_a_agg.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for est in agg["estimator"].unique():
        sub = agg[agg["estimator"] == est]
        axes[0].plot(sub["V"], sub["bias"], marker="o", label=est)
        axes[1].plot(sub["V"], sub["rmse"], marker="o", label=est)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("V"); axes[0].set_ylabel("Bias of π̂"); axes[0].legend()
    axes[1].set_xlabel("V"); axes[1].set_ylabel("RMSE of π̂"); axes[1].legend()
    fig.suptitle("Experiment A: bias and RMSE vs vocabulary size")
    fig.tight_layout()
    fig.savefig(OUT / "fig_a.pdf")
    print(f"Saved results to {OUT}")


if __name__ == "__main__":
    run()
