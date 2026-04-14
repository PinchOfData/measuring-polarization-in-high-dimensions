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


def _normalize_rows(counts: sp.spmatrix, mode: str) -> sp.csr_matrix:
    """Row-wise normalization, preserving sparsity."""
    C = counts if sp.isspmatrix_csr(counts) else counts.tocsr()
    if mode == "count":
        return C.astype(float)
    if mode == "binary":
        out = C.copy().astype(float)
        out.data[:] = 1.0
        return out
    if mode == "freq":
        row_sums = np.asarray(C.sum(axis=1)).ravel()
        safe = np.where(row_sums > 0, row_sums, 1.0)
        return (sp.diags(1.0 / safe) @ C.astype(float)).tocsr()
    raise ValueError(f"unknown normalize={mode!r}")


def _choice_probs_np(
    alpha_t: np.ndarray,   # (V,)
    gamma: np.ndarray,     # (V, P)
    phi_t: np.ndarray,     # (V,)
    X_row: np.ndarray,     # (P,)
    party: float,
) -> np.ndarray:
    """Numpy replica of partisanship.choice_probs to avoid torch roundtrips."""
    u = alpha_t + X_row @ gamma.T + phi_t * party
    u = u - u.max()
    e = np.exp(u)
    return e / e.sum()


def _rho_np(
    alpha_t: np.ndarray,
    gamma: np.ndarray,
    phi_t: np.ndarray,
    X_row: np.ndarray,
    eps: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_R = _choice_probs_np(alpha_t, gamma, phi_t, X_row, 1.0)
    q_D = _choice_probs_np(alpha_t, gamma, phi_t, X_row, 0.0)
    rho = q_R / (q_R + q_D + eps)
    return q_R, q_D, rho


def scale_documents(
    estimator,
    counts_matrix: sp.spmatrix,
    session,
    normalize: str = "freq",
) -> np.ndarray:
    """Vectorized media_slant eq. (1): s_m = Σ_b f_bm · φ̂_{b, t_m}.

    Returns a dense ``(M,)`` numpy array of per-document real-valued scores.
    """
    _check_vocab(estimator, counts_matrix)
    M = counts_matrix.shape[0]
    F = _normalize_rows(counts_matrix, normalize)

    phi = estimator.phi_  # (V, T)

    if np.ndim(session) == 0:
        t = int(session)
        return np.asarray(F @ phi[:, t]).ravel()

    sess = np.asarray(session, dtype=int)
    if sess.shape[0] != M:
        raise ValueError(
            f"session array length {sess.shape[0]} does not match "
            f"counts_matrix.shape[0]={M}"
        )

    out = np.zeros(M, dtype=float)
    for t in np.unique(sess):
        rows = np.where(sess == t)[0]
        F_sub = F[rows]
        out[rows] = np.asarray(F_sub @ phi[:, int(t)]).ravel()
    return out


def scale_document(
    estimator,
    counts_new: sp.spmatrix,
    session: int,
    normalize: str = "freq",
) -> float:
    """Media_slant eq. (1): dot-product of document freqs with phi for session t.

    Thin wrapper around ``scale_documents`` for a single-row input.
    """
    _check_vocab(estimator, counts_new)
    scores = scale_documents(estimator, counts_new, session=session, normalize=normalize)
    return float(scores[0])


def score_documents(
    estimator,
    counts_matrix: sp.spmatrix,
    session,
    X_new: np.ndarray | None = None,
    normalize: str = "freq",
) -> dict:
    """Vectorized per-document posterior π.

    For doc m:  π_m = Σ_b f_bm · ρ_b(t_m, x_m)
    where ρ_b(t, x) = q_R_b(t, x) / (q_R_b(t, x) + q_D_b(t, x)).

    Returns ``{"pi": np.ndarray of shape (M,)}``.
    """
    _check_vocab(estimator, counts_matrix)
    M = counts_matrix.shape[0]
    P = int(estimator.n_covariates_)

    # Canonicalize X_new to a (M_like, P) numpy array and a boolean flag
    # indicating whether it's constant across documents.
    if X_new is None:
        X_arr = np.zeros((1, P), dtype=float)
        x_constant = True
    else:
        X_arr = np.asarray(X_new, dtype=float)
        if X_arr.ndim == 1:
            if X_arr.shape[0] != P:
                raise ValueError(
                    f"X_new has length {X_arr.shape[0]} but estimator expects P={P}"
                )
            X_arr = X_arr.reshape(1, P)
            x_constant = True
        elif X_arr.ndim == 2:
            if X_arr.shape[1] != P:
                raise ValueError(
                    f"X_new has {X_arr.shape[1]} columns but estimator expects P={P}"
                )
            if X_arr.shape[0] == 1:
                x_constant = True
            elif X_arr.shape[0] == M:
                x_constant = False
            else:
                raise ValueError(
                    f"X_new has {X_arr.shape[0]} rows but counts_matrix has {M} rows"
                )
        else:
            raise ValueError(f"X_new must be 1D or 2D, got ndim={X_arr.ndim}")

    # Canonicalize session
    if np.ndim(session) == 0:
        sess = np.full(M, int(session), dtype=int)
        s_constant = True
    else:
        sess = np.asarray(session, dtype=int)
        if sess.shape[0] != M:
            raise ValueError(
                f"session array length {sess.shape[0]} does not match "
                f"counts_matrix.shape[0]={M}"
            )
        s_constant = len(np.unique(sess)) == 1

    F = _normalize_rows(counts_matrix, normalize)

    alpha = estimator.alpha_  # (V, T)
    phi = estimator.phi_      # (V, T)
    gamma = estimator.gamma_  # (V, P)

    out = np.zeros(M, dtype=float)

    # Fast path: single (t, x) group for the whole batch.
    if s_constant and x_constant:
        t = int(sess[0])
        x_row = X_arr[0]
        _, _, rho = _rho_np(alpha[:, t], gamma, phi[:, t], x_row)
        return {"pi": np.asarray(F @ rho).ravel()}

    # General path: group rows by (t, x_tuple).
    # Build grouping keys.
    if x_constant:
        # X is the same for everyone; group by session only.
        x_row = X_arr[0]
        for t in np.unique(sess):
            rows = np.where(sess == t)[0]
            _, _, rho = _rho_np(alpha[:, int(t)], gamma, phi[:, int(t)], x_row)
            out[rows] = np.asarray(F[rows] @ rho).ravel()
        return {"pi": out}

    # Per-doc X: group by (t, bytes(x_row)).
    # Make sure X row bytes are canonicalized (contiguous float64).
    X_c = np.ascontiguousarray(X_arr, dtype=np.float64)
    groups: dict[tuple[int, bytes], list[int]] = {}
    for i in range(M):
        key = (int(sess[i]), X_c[i].tobytes())
        groups.setdefault(key, []).append(i)

    for (t, _xb), row_list in groups.items():
        rows = np.asarray(row_list, dtype=int)
        x_row = X_c[rows[0]]
        _, _, rho = _rho_np(alpha[:, t], gamma, phi[:, t], x_row)
        out[rows] = np.asarray(F[rows] @ rho).ravel()

    return {"pi": out}


def score_document(
    estimator,
    counts_new: sp.spmatrix,
    session: int,
    X_new: np.ndarray | None = None,
    normalize: str = "freq",
) -> dict:
    """Posterior π for a single doc treated as a hypothetical speaker.

    π = Σ_b f_b · ρ_b(t, x)   — the true per-document posterior, not the
    population-level average.
    """
    _check_vocab(estimator, counts_new)
    t = int(session)
    P = int(estimator.n_covariates_)
    if X_new is None:
        X_row = np.zeros(P, dtype=float)
    else:
        X_row = np.asarray(X_new, dtype=float).ravel()
        if X_row.shape[0] != P:
            raise ValueError(
                f"X_new has length {X_row.shape[0]} but estimator expects P={P}"
            )

    alpha_t = estimator.alpha_[:, t]
    phi_t = estimator.phi_[:, t]
    gamma = estimator.gamma_

    q_R, q_D, rho = _rho_np(alpha_t, gamma, phi_t, X_row)

    F = _normalize_rows(counts_new, normalize)
    f_b = np.asarray(F.toarray()).ravel()
    pi = float((f_b * rho).sum())

    return {
        "pi": pi,
        "rho": rho,
        "q_R": q_R,
        "q_D": q_D,
    }
