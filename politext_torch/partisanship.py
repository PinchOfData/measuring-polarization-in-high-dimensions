"""Partisanship primitives: model-based and leave-out estimators."""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import torch


def choice_probs(
    alpha_t: torch.Tensor,         # (V,)
    gamma: torch.Tensor,           # (V, P)
    phi_t: torch.Tensor,           # (V,)
    X_row: torch.Tensor,           # (P,)
    party: float,
) -> torch.Tensor:
    u = alpha_t + X_row @ gamma.T + phi_t * party
    return torch.softmax(u, dim=-1)


def posterior_rho(
    alpha_t: torch.Tensor,
    gamma: torch.Tensor,
    phi_t: torch.Tensor,
    X_row: torch.Tensor,
    eps: float = 1e-30,
) -> torch.Tensor:
    q_R = choice_probs(alpha_t, gamma, phi_t, X_row, 1.0)
    q_D = choice_probs(alpha_t, gamma, phi_t, X_row, 0.0)
    return q_R / (q_R + q_D + eps)


def partisanship(
    alpha: torch.Tensor,           # (V, T)
    gamma: torch.Tensor,           # (V, P)
    phi: torch.Tensor,             # (V, T)
    X: torch.Tensor,               # (N, P)
    session: torch.Tensor,         # (N,)
    party: torch.Tensor,           # (N,) in {0, 1}
) -> torch.Tensor:
    """Session-level average partisanship (politext eq. 3-5)."""
    T = alpha.shape[1]
    N = X.shape[0]
    out = torch.full((T,), float("nan"), dtype=torch.float64)
    for t in range(T):
        mask = (session == t)
        idx = torch.nonzero(mask, as_tuple=False).ravel()
        if idx.numel() == 0:
            continue
        has_R = (party[idx] == 1).any()
        has_D = (party[idx] == 0).any()
        if not (has_R and has_D):
            warnings.warn(f"Session {t}: missing at least one party; "
                          "partisanship is undefined.", UserWarning, stacklevel=2)
            continue
        total = 0.0
        for i in idx:
            Xi = X[i]
            q_R = choice_probs(alpha[:, t], gamma, phi[:, t], Xi, 1.0)
            q_D = choice_probs(alpha[:, t], gamma, phi[:, t], Xi, 0.0)
            rho = q_R / (q_R + q_D + 1e-30)
            pi_i = 0.5 * (q_R * rho).sum() + 0.5 * (q_D * (1 - rho)).sum()
            total = total + pi_i
        out[t] = (total / idx.numel()).item()
    return out


def leave_out_partisanship(
    counts: sp.csr_matrix,
    party: np.ndarray,
    session: np.ndarray,
    speaker_id: np.ndarray | None = None,
    eps: float = 1e-30,
) -> np.ndarray:
    """Politext eq. (8) leave-one-speaker-out estimator.

    Closed form: for speaker i in session t, use q_hat_i (i's own empirical freq)
    paired with rho_hat_{-i,t} (leave-one-out ρ).
    """
    N, V = counts.shape
    if speaker_id is None:
        speaker_id = np.arange(N)
    party = np.asarray(party).astype(float)
    session = np.asarray(session)

    T = int(session.max()) + 1 if N > 0 else 0
    out = np.full(T, np.nan)

    counts_dense = counts.toarray().astype(np.float64)
    m = counts_dense.sum(axis=1)

    for t in range(T):
        in_t = np.where(session == t)[0]
        if len(in_t) == 0:
            continue
        mask_R = in_t[party[in_t] == 1.0]
        mask_D = in_t[party[in_t] == 0.0]
        if len(mask_R) == 0 or len(mask_D) == 0:
            continue

        sum_R = counts_dense[mask_R].sum(axis=0)   # (V,)
        sum_D = counts_dense[mask_D].sum(axis=0)
        m_R = m[mask_R].sum()
        m_D = m[mask_D].sum()

        # Group counts by speaker (for leaving out by unique speaker_id).
        sp_in_R = speaker_id[mask_R]
        sp_in_D = speaker_id[mask_D]
        # Map: for each speaker id, precompute their total counts and m.
        # With speaker_id = arange(N) (default), each row *is* one speaker,
        # so counts_dense[i] equals their total.
        # For multi-row speakers we need to aggregate first:
        unique_R, inv_R = np.unique(sp_in_R, return_inverse=True)
        unique_D, inv_D = np.unique(sp_in_D, return_inverse=True)
        speaker_counts_R = np.zeros((len(unique_R), V))
        speaker_m_R = np.zeros(len(unique_R))
        for k, idx in enumerate(mask_R):
            speaker_counts_R[inv_R[k]] += counts_dense[idx]
            speaker_m_R[inv_R[k]] += m[idx]
        speaker_counts_D = np.zeros((len(unique_D), V))
        speaker_m_D = np.zeros(len(unique_D))
        for k, idx in enumerate(mask_D):
            speaker_counts_D[inv_D[k]] += counts_dense[idx]
            speaker_m_D[inv_D[k]] += m[idx]

        # Leave-one-speaker-out ρ for R speakers
        def loo_pi_for_party(party_speakers_counts, party_speakers_m,
                             sum_own, m_own, sum_other, m_other, is_R: bool):
            num = 0.0
            for k in range(len(party_speakers_counts)):
                exc = sum_own - party_speakers_counts[k]
                exc_m = m_own - party_speakers_m[k]
                if exc_m <= 0:
                    continue
                qhat_own = exc / exc_m
                qhat_other = sum_other / m_other if m_other > 0 else np.zeros(V)
                denom = qhat_own + qhat_other
                if is_R:
                    rho = np.where(denom > 0, qhat_own / (denom + eps), 0.5)
                else:
                    rho = np.where(denom > 0, qhat_other / (denom + eps), 0.5)
                # speaker own empirical frequency
                if party_speakers_m[k] <= 0:
                    continue
                q_i = party_speakers_counts[k] / party_speakers_m[k]
                if is_R:
                    num += (q_i * rho).sum()
                else:
                    num += (q_i * (1 - rho)).sum()
            return num / max(len(party_speakers_counts), 1)

        term_R = loo_pi_for_party(
            speaker_counts_R, speaker_m_R,
            sum_own=sum_R, m_own=m_R,
            sum_other=sum_D, m_other=m_D,
            is_R=True,
        )
        term_D = loo_pi_for_party(
            speaker_counts_D, speaker_m_D,
            sum_own=sum_D, m_own=m_D,
            sum_other=sum_R, m_other=m_R,
            is_R=False,
        )
        out[t] = 0.5 * term_R + 0.5 * term_D

    return out
