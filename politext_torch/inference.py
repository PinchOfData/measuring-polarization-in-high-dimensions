"""Subsampling-based confidence intervals (Politis-Romano-Wolf)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp


def subsample_ci(
    estimator_factory: Callable,
    counts: sp.csr_matrix,
    party: np.ndarray,
    session: np.ndarray,
    X: np.ndarray | None = None,
    speaker_id: np.ndarray | None = None,
    n_subsamples: int = 100,
    frac: float = 0.1,
    alpha: float = 0.05,
    transform: str = "log",
    seed: int | None = None,
    n_jobs: int = 1,
) -> dict:
    """Speaker-level subsampling CI for session-level partisanship.

    Procedure (Politis-Romano-Wolf 1999, Thm 2.2.1):
      1. Full-sample fit -> pi_hat.
      2. Draw n_subsamples subsamples of speakers (size = round(frac * n_speakers))
         without replacement.
      3. Fit estimator_factory() on each subsample -> pi_b.
      4. Compute sqrt(tau_b) * (g(pi_b) - g(pi_hat)) quantiles, invert to get CI.

    transform: "identity" or "log" (default). Log uses g(x) = log(x - 0.5),
    which respects the [0.5, 1] support and matches politext Figure 1.
    """
    N = counts.shape[0]
    party = np.asarray(party)
    session = np.asarray(session)
    if speaker_id is None:
        speaker_id = np.arange(N)
    else:
        speaker_id = np.asarray(speaker_id)

    unique_speakers = np.unique(speaker_id)
    n_speakers = len(unique_speakers)
    n_sub_spk = max(1, int(round(frac * n_speakers)))

    # Full-sample fit
    est = estimator_factory()
    fit_kwargs = dict(party=party, session=session, speaker_id=speaker_id)
    if X is not None:
        fit_kwargs["X"] = X
    est.fit(counts, **fit_kwargs)
    pi_hat = np.asarray(est.partisanship_, dtype=float)
    T = pi_hat.shape[0]

    rng = np.random.default_rng(seed)

    def _run_one_subsample(b: int) -> np.ndarray:
        sub_rng = np.random.default_rng(None if seed is None else seed + 1 + b)
        chosen_speakers = sub_rng.choice(unique_speakers, size=n_sub_spk, replace=False)
        row_mask = np.isin(speaker_id, chosen_speakers)
        rows = np.where(row_mask)[0]
        sub_counts = counts[rows]
        sub_party = party[rows]
        sub_session = session[rows]
        sub_speaker_id = speaker_id[rows]
        sub_X = X[rows] if X is not None else None
        est_b = estimator_factory()
        kw = dict(party=sub_party, session=sub_session, speaker_id=sub_speaker_id)
        if sub_X is not None:
            kw["X"] = sub_X
        est_b.fit(sub_counts, **kw)
        pi_b = np.asarray(est_b.partisanship_, dtype=float)
        # Pad with NaN for missing sessions
        if pi_b.shape[0] < T:
            padded = np.full(T, np.nan)
            padded[: pi_b.shape[0]] = pi_b
            pi_b = padded
        return pi_b

    if n_jobs == 1:
        subs = np.stack([_run_one_subsample(b) for b in range(n_subsamples)], axis=0)
    else:
        from joblib import Parallel, delayed
        subs = np.stack(
            Parallel(n_jobs=n_jobs)(delayed(_run_one_subsample)(b)
                                    for b in range(n_subsamples)),
            axis=0,
        )

    tau = float(n_sub_spk)
    n_full = float(n_speakers)

    def _g(x: np.ndarray) -> np.ndarray:
        if transform == "identity":
            return x
        return np.log(np.maximum(x - 0.5, 1e-12))

    def _g_inv(u: np.ndarray) -> np.ndarray:
        if transform == "identity":
            return u
        return 0.5 + np.exp(u)

    # Per-session: compute quantiles of Q_b = sqrt(tau) * (g(pi_b) - g(pi_hat))
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    for t in range(T):
        if not np.isfinite(pi_hat[t]):
            continue
        valid = np.isfinite(subs[:, t])
        if valid.sum() < 2:
            continue
        g_hat = _g(np.array([pi_hat[t]]))[0]
        use_log_here = (transform == "log") and (pi_hat[t] - 0.5 > 1e-6)
        if transform == "log" and not use_log_here:
            # Fall back to identity per session near 0.5.
            Q = np.sqrt(tau) * (subs[valid, t] - pi_hat[t])
            q_lo, q_hi = np.quantile(Q, [alpha / 2, 1 - alpha / 2])
            lo[t] = pi_hat[t] - q_hi / np.sqrt(n_full)
            hi[t] = pi_hat[t] - q_lo / np.sqrt(n_full)
        else:
            Q = np.sqrt(tau) * (_g(subs[valid, t]) - g_hat)
            q_lo, q_hi = np.quantile(Q, [alpha / 2, 1 - alpha / 2])
            lo[t] = _g_inv(g_hat - q_hi / np.sqrt(n_full))
            hi[t] = _g_inv(g_hat - q_lo / np.sqrt(n_full))

    return {
        "estimate": pi_hat,
        "ci_lower": lo,
        "ci_upper": hi,
        "subsample_estimates": subs,
        "n_sub": n_sub_spk,
        "n_full": n_speakers,
        "frac": frac,
    }
