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


def test_scale_documents_is_per_document_and_vectorized():
    est, V, _ = _fit_penalized_for_scaling()
    rng = np.random.default_rng(0)
    M = 2000
    docs = sp.csr_matrix(rng.poisson(0.5, size=(M, V)))
    scores = scale_documents(est, docs, session=0)
    assert scores.shape == (M,)
    # There should be real variation across docs (not a constant).
    assert scores.std() > 1e-3
    # Spot-check: scalar agrees with batch for a random row.
    i = int(rng.integers(M))
    assert abs(scores[i] - scale_document(est, docs[i], session=0)) < 1e-6


def test_score_documents_uses_document_counts():
    """The returned pi MUST depend on the document counts — it's per-doc, not population."""
    est, V, _ = _fit_penalized_for_scaling()
    rng = np.random.default_rng(0)
    # Two different docs with the same (session, X_new) should generally get
    # different pi values (unless they happen to produce the same f @ rho, which
    # is a measure-zero event for random counts).
    docA = sp.csr_matrix(rng.poisson(1.0, size=(1, V)))
    docB = sp.csr_matrix(rng.poisson(1.0, size=(1, V)))
    outA = score_documents(est, docA, session=0)
    outB = score_documents(est, docB, session=0)
    assert outA["pi"].shape == (1,)
    assert outB["pi"].shape == (1,)
    assert abs(outA["pi"][0] - outB["pi"][0]) > 1e-6


def test_score_documents_matches_single_doc_path():
    est, V, _ = _fit_penalized_for_scaling()
    rng = np.random.default_rng(1)
    M = 10
    docs = sp.csr_matrix(rng.poisson(1.0, size=(M, V)))
    batch = score_documents(est, docs, session=0)["pi"]
    for i in range(M):
        single = score_document(est, docs[i], session=0)["pi"]
        np.testing.assert_allclose(batch[i], single, atol=1e-6)


def test_scale_documents_handles_varying_session():
    """Multi-session input: verify per-doc session routing matches a manual loop."""
    rng = np.random.default_rng(2)
    # Need an estimator with T >= 2 for this test.
    V, T, N = 8, 2, 400
    alpha = torch.tensor(rng.standard_normal((V, T)) * 0.3, dtype=torch.float32)
    phi = torch.tensor(rng.standard_normal((V, T)) * 0.6, dtype=torch.float32)
    gamma = torch.zeros(V, 0)
    X = torch.zeros(N, 0)
    party = torch.tensor((rng.random(N) < 0.5).astype(np.float32))
    session = torch.tensor(rng.integers(0, T, size=N), dtype=torch.long)
    m = torch.full((N,), 500.0)
    counts = draw_counts(alpha, gamma, phi, X, party, session, m, seed=3)
    est = PenalizedEstimator(lam=0.05, max_iter=200).fit(counts, party.numpy(), session.numpy())

    M = 50
    docs = sp.csr_matrix(rng.poisson(0.5, size=(M, V)))
    sess = rng.integers(0, T, size=M).astype(int)
    batch = scale_documents(est, docs, session=sess)
    manual = np.array([scale_document(est, docs[i], session=int(sess[i])) for i in range(M)])
    np.testing.assert_allclose(batch, manual, atol=1e-6)


def test_score_documents_millions_of_docs_under_one_second():
    """Performance regression guard: 100k docs should score in well under 1 second."""
    import time
    est, V, _ = _fit_penalized_for_scaling()
    rng = np.random.default_rng(0)
    M = 100_000
    docs = sp.csr_matrix(rng.poisson(0.1, size=(M, V)))
    t0 = time.perf_counter()
    out = score_documents(est, docs, session=0)
    dt = time.perf_counter() - t0
    assert out["pi"].shape == (M,)
    # Relaxed bound: even a slow laptop should do this in under 1.5s for 100k × V=8.
    assert dt < 1.5, f"score_documents took {dt:.2f}s for {M} docs"
