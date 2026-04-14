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


def _true_partisanship(alpha, gamma, phi, X, session, party):
    from politext_torch.partisanship import partisanship
    return partisanship(alpha, gamma, phi, X, session, party).numpy()


def make_mc_A(V: int, T: int = 5, N: int = 1000, P: int = 0,
              m_value: float = 100.0, seed: int = 0) -> dict:
    """Bias/RMSE experiment DGP: covariate-free multinomial with moderate phi."""
    rng = np.random.default_rng(seed)
    alpha = torch.tensor(rng.standard_normal((V, T)) * 0.3, dtype=torch.float32)
    phi = torch.tensor(rng.standard_normal((V, T)) * 0.5, dtype=torch.float32)
    gamma = torch.zeros(V, P)
    party = torch.tensor((rng.random(N) < 0.5).astype(np.float32))
    session = torch.tensor(rng.integers(0, T, size=N), dtype=torch.long)
    X = torch.zeros(N, P)
    m = torch.full((N,), m_value)
    counts = draw_counts(alpha, gamma, phi, X, party, session, m, seed=seed + 17)
    true_pi = _true_partisanship(alpha, gamma, phi, X, session, party)
    return dict(
        counts=counts, party=party.numpy(), session=session.numpy(),
        X=None if P == 0 else X.numpy(), true_pi=true_pi,
        true_alpha=alpha, true_gamma=gamma, true_phi=phi,
    )


def make_mc_B(V: int = 200, T: int = 10, N: int = 1000, seed: int = 0) -> dict:
    """Coverage experiment DGP: identical form to A; kept separate so
    experiments can pin their own sizes."""
    return make_mc_A(V=V, T=T, N=N, P=0, m_value=200.0, seed=seed)


def make_mc_C(V: int, T: int = 5, N: int = 1000, seed: int = 0) -> dict:
    """Null experiment: phi = 0, so true partisanship equals 0.5."""
    rng = np.random.default_rng(seed)
    alpha = torch.tensor(rng.standard_normal((V, T)) * 0.3, dtype=torch.float32)
    phi = torch.zeros(V, T)
    gamma = torch.zeros(V, 0)
    party = torch.tensor((rng.random(N) < 0.5).astype(np.float32))
    session = torch.tensor(rng.integers(0, T, size=N), dtype=torch.long)
    X = torch.zeros(N, 0)
    m = torch.full((N,), 100.0)
    counts = draw_counts(alpha, gamma, phi, X, party, session, m, seed=seed + 17)
    true_pi = np.full(T, 0.5)  # by construction
    return dict(
        counts=counts, party=party.numpy(), session=session.numpy(),
        X=None, true_pi=true_pi,
        true_alpha=alpha, true_gamma=gamma, true_phi=phi,
    )
