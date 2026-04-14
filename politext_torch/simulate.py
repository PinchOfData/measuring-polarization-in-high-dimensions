"""Synthetic data generation for the phrase-choice Poisson/multinomial model."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def draw_counts(
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    phi: torch.Tensor,
    X: torch.Tensor,
    party: torch.Tensor,
    session: torch.Tensor,
    m: torch.Tensor,
    seed: int | None = None,
) -> sp.csr_matrix:
    """Sample phrase counts from the Gentzkow-Shapiro-Taddy (2019) model.

    For each speaker row i, compute u_ij = alpha[j, t_i] + X[i] @ gamma[j]
      + phi[j, t_i] * K_i, softmax to q_ij, then draw c_i ~ Multinomial(m_i, q_i).

    Parameters
    ----------
    alpha : (V, T)
    gamma : (V, P)
    phi   : (V, T)
    X     : (N, P)
    party : (N,) in {0, 1}
    session : (N,) long in 0..T-1
    m     : (N,) verbosity
    seed  : RNG seed.

    Returns
    -------
    scipy.sparse.csr_matrix of shape (N, V) with integer counts.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    V = alpha.shape[0]

    alpha_i = alpha[:, session].T               # (N, V)
    phi_i = phi[:, session].T                   # (N, V)
    u = alpha_i + X @ gamma.T + phi_i * party[:, None]   # (N, V)
    q = torch.softmax(u, dim=-1).cpu().numpy()  # (N, V)
    m_np = m.cpu().numpy().astype(np.int64)

    rows, cols, vals = [], [], []
    for i in range(N):
        c_i = rng.multinomial(m_np[i], q[i])
        nz = np.flatnonzero(c_i)
        rows.extend([i] * len(nz))
        cols.extend(nz.tolist())
        vals.extend(c_i[nz].tolist())

    return sp.csr_matrix((vals, (rows, cols)), shape=(N, V), dtype=np.int64)
