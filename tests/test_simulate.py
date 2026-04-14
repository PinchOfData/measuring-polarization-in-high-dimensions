import numpy as np
import scipy.sparse as sp
import torch
from politext_torch.simulate import draw_counts


def test_draw_counts_returns_csr_with_right_shape(tiny_dgp):
    d = tiny_dgp
    counts = draw_counts(
        alpha=d["alpha"], gamma=d["gamma"], phi=d["phi"],
        X=d["X"], party=d["party"], session=d["session"], m=d["m"],
        seed=42,
    )
    assert isinstance(counts, sp.csr_matrix)
    assert counts.shape == (d["N"], d["V"])
    # verbosity preserved: row sums equal m
    np.testing.assert_array_equal(np.asarray(counts.sum(axis=1)).ravel(),
                                  d["m"].numpy())


def test_draw_counts_reproducible_with_seed(tiny_dgp):
    d = tiny_dgp
    a = draw_counts(d["alpha"], d["gamma"], d["phi"], d["X"],
                    d["party"], d["session"], d["m"], seed=7)
    b = draw_counts(d["alpha"], d["gamma"], d["phi"], d["X"],
                    d["party"], d["session"], d["m"], seed=7)
    np.testing.assert_array_equal(a.toarray(), b.toarray())


def test_draw_counts_freqs_converge_to_softmax_probs(tiny_dgp):
    """Law of large numbers: empirical freqs converge to softmax(u) as m grows."""
    d = tiny_dgp
    # single speaker, huge verbosity
    alpha_t = d["alpha"][:, 0]
    phi_t = d["phi"][:, 0]
    u = alpha_t + d["X"][0] @ d["gamma"].T + phi_t * d["party"][0]
    expected = torch.softmax(u, dim=-1).numpy()

    m = torch.tensor([1_000_000.0])
    counts = draw_counts(
        alpha=d["alpha"], gamma=d["gamma"], phi=d["phi"],
        X=d["X"][:1], party=d["party"][:1], session=d["session"][:1], m=m,
        seed=1,
    )
    empirical = counts.toarray()[0] / counts.sum()
    np.testing.assert_allclose(empirical, expected, atol=2e-3)


def test_make_mc_A_shapes_sensible():
    from politext_torch.simulate import make_mc_A
    out = make_mc_A(V=50, T=3, N=200, seed=0)
    assert out["counts"].shape == (200, 50)
    assert out["party"].shape == (200,)
    assert out["session"].shape == (200,)
    assert out["true_pi"].shape == (3,)


def test_make_mc_C_is_null():
    from politext_torch.simulate import make_mc_C
    out = make_mc_C(V=20, T=2, N=100, seed=0)
    # phi = 0 -> true partisanship is exactly 0.5 in every session.
    assert out["true_phi"].abs().max() < 1e-9
    assert np.allclose(out["true_pi"], 0.5)
