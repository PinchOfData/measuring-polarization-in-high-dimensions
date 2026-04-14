import numpy as np
import pytest
import scipy.sparse as sp
import torch
from politext_torch.estimators import PenalizedEstimator
from politext_torch.scale import (
    scale_document, scale_documents,
    score_document, score_documents,
)
from politext_torch.simulate import draw_counts


def _fit_penalized_for_scaling(seed=0):
    rng = np.random.default_rng(seed)
    V, T, N = 8, 1, 400
    alpha = torch.tensor(rng.standard_normal((V, T)) * 0.3, dtype=torch.float32)
    phi = torch.tensor(rng.standard_normal((V, T)) * 0.6, dtype=torch.float32)
    gamma = torch.zeros(V, 0)
    X = torch.zeros(N, 0)
    party = torch.tensor((rng.random(N) < 0.5).astype(np.float32))
    session = torch.zeros(N, dtype=torch.long)
    m = torch.full((N,), 500.0)
    counts = draw_counts(alpha, gamma, phi, X, party, session, m, seed=seed + 1)
    est = PenalizedEstimator(lam=0.05, max_iter=300)
    est.fit(counts, party.numpy(), session.numpy())
    return est, V, T


def test_scale_document_equals_dot_product_of_freqs_and_phi():
    est, V, _ = _fit_penalized_for_scaling()
    doc = sp.csr_matrix(np.array([[3, 0, 1, 0, 0, 2, 0, 4]]))
    score = scale_document(est, doc, session=0)
    freqs = np.array([3, 0, 1, 0, 0, 2, 0, 4], dtype=float)
    freqs = freqs / freqs.sum()
    expected = float((freqs * est.phi_[:, 0]).sum())
    np.testing.assert_allclose(score, expected, atol=1e-6)


def test_scale_document_rejects_wrong_vocab():
    est, V, _ = _fit_penalized_for_scaling()
    bad = sp.csr_matrix(np.ones((1, V + 3)))
    with pytest.raises(ValueError, match="vocab"):
        scale_document(est, bad, session=0)


def test_scale_documents_batches_and_matches_scalars():
    est, V, _ = _fit_penalized_for_scaling()
    docs = sp.csr_matrix(np.array([[3, 0, 1, 0, 0, 2, 0, 4],
                                   [0, 5, 0, 2, 0, 0, 1, 0]]))
    batch = scale_documents(est, docs, session=0)
    individual = np.array([
        scale_document(est, docs[0], session=0),
        scale_document(est, docs[1], session=0),
    ])
    np.testing.assert_allclose(batch, individual, atol=1e-6)


def test_score_document_returns_pi_rho_q():
    est, V, _ = _fit_penalized_for_scaling()
    doc = sp.csr_matrix(np.array([[1, 1, 0, 0, 0, 0, 0, 0]]))
    out = score_document(est, doc, session=0)
    assert set(out) >= {"pi", "rho", "q_R", "q_D"}
    assert 0 <= out["pi"] <= 1
    assert out["rho"].shape == (V,)
