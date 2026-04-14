"""Scaling new unseen documents with a fitted PenalizedEstimator."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from politext_torch.partisanship import choice_probs, posterior_rho


def _check_vocab(est, counts: sp.csr_matrix) -> None:
    if counts.shape[1] != est.vocab_size_:
        raise ValueError(
            f"vocab size mismatch: estimator trained on V={est.vocab_size_} "
            f"but received counts with {counts.shape[1]} columns. "
            "Use the same fitted CountVectorizer's `.transform(...)` at both "
            "training and scoring time."
        )


def _freqs(counts_row: sp.csr_matrix, normalize: str) -> np.ndarray:
    arr = np.asarray(counts_row.toarray()).ravel().astype(float)
    if normalize == "count":
        return arr
    if normalize == "binary":
        return (arr > 0).astype(float)
    total = arr.sum()
    if total == 0:
        return arr
    return arr / total


def scale_document(
    estimator,
    counts_new: sp.csr_matrix,
    session: int,
    normalize: str = "freq",
) -> float:
    """Media_slant eq. (1): dot-product of document freqs with phi for session t."""
    _check_vocab(estimator, counts_new)
    f_b = _freqs(counts_new, normalize)
    phi_t = estimator.phi_[:, int(session)]
    return float((f_b * phi_t).sum())


def scale_documents(
    estimator,
    counts_matrix: sp.csr_matrix,
    session,
    normalize: str = "freq",
) -> np.ndarray:
    _check_vocab(estimator, counts_matrix)
    M = counts_matrix.shape[0]
    if np.ndim(session) == 0:
        sess = np.full(M, int(session))
    else:
        sess = np.asarray(session, dtype=int)

    out = np.zeros(M)
    for m in range(M):
        f_b = _freqs(counts_matrix[m], normalize)
        phi_t = estimator.phi_[:, sess[m]]
        out[m] = float((f_b * phi_t).sum())
    return out


def score_document(
    estimator,
    counts_new: sp.csr_matrix,
    session: int,
    X_new: np.ndarray | None = None,
    normalize: str = "freq",
) -> dict:
    """Posterior π for a doc treated as a hypothetical speaker (politext eq. 3-4)."""
    _check_vocab(estimator, counts_new)
    t = int(session)
    alpha_t = torch.from_numpy(estimator.alpha_[:, t]).float()
    phi_t = torch.from_numpy(estimator.phi_[:, t]).float()
    gamma = torch.from_numpy(estimator.gamma_).float()
    if X_new is None:
        X_row = torch.zeros(estimator.n_covariates_)
    else:
        X_row = torch.as_tensor(X_new, dtype=torch.float32).ravel()

    q_R = choice_probs(alpha_t, gamma, phi_t, X_row, 1.0)
    q_D = choice_probs(alpha_t, gamma, phi_t, X_row, 0.0)
    rho = posterior_rho(alpha_t, gamma, phi_t, X_row)

    f_b = _freqs(counts_new, normalize)
    pi = float(0.5 * (f_b * rho.numpy()).sum() + 0.5 * (f_b * (1 - rho.numpy())).sum())
    # Note: the above reduces to 0.5 for uniform f_b — correct only when we
    # pair rho with q_R (for R-hypothesis) and 1-rho with q_D (for D-hypothesis):
    pi = float(
        0.5 * (q_R.numpy() * rho.numpy()).sum()
        + 0.5 * (q_D.numpy() * (1 - rho.numpy())).sum()
    )
    return {
        "pi": pi,
        "rho": rho.numpy(),
        "q_R": q_R.numpy(),
        "q_D": q_D.numpy(),
    }


def score_documents(
    estimator,
    counts_matrix: sp.csr_matrix,
    session,
    X_new: np.ndarray | None = None,
    normalize: str = "freq",
) -> dict:
    _check_vocab(estimator, counts_matrix)
    M = counts_matrix.shape[0]
    if np.ndim(session) == 0:
        sess = np.full(M, int(session))
    else:
        sess = np.asarray(session, dtype=int)

    pi_arr = np.zeros(M)
    for m in range(M):
        x = None if X_new is None else X_new[m]
        pi_arr[m] = score_document(estimator, counts_matrix[m],
                                   session=sess[m], X_new=x, normalize=normalize)["pi"]
    return {"pi": pi_arr}
