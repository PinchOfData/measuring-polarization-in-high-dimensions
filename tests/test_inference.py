import numpy as np
import torch
from politext_torch.estimators import LeaveOutEstimator
from politext_torch.inference import subsample_ci
from politext_torch.simulate import draw_counts


def _dgp_for_ci(N=400, V=5, T=2, seed=0):
    rng = np.random.default_rng(seed)
    alpha = torch.tensor(rng.standard_normal((V, T)) * 0.3, dtype=torch.float32)
    phi = torch.tensor(rng.standard_normal((V, T)) * 0.4, dtype=torch.float32)
    gamma = torch.zeros(V, 0)
    X = torch.zeros(N, 0)
    party = torch.tensor((rng.random(N) < 0.5).astype(np.float32))
    session = torch.tensor(rng.integers(0, T, size=N), dtype=torch.long)
    m = torch.full((N,), 500.0)
    counts = draw_counts(alpha, gamma, phi, X, party, session, m, seed=seed + 1)
    return counts, party.numpy(), session.numpy(), X.numpy()


def test_subsample_ci_shapes_and_point_estimate_inside_ci():
    counts, party, session, X = _dgp_for_ci()
    factory = lambda: LeaveOutEstimator()
    result = subsample_ci(
        factory, counts, party, session,
        n_subsamples=20, frac=0.3, seed=1, transform="identity",
    )
    T = 2
    assert result["estimate"].shape == (T,)
    assert result["ci_lower"].shape == (T,)
    assert result["ci_upper"].shape == (T,)
    assert result["subsample_estimates"].shape == (20, T)
    # Point estimate lies between the bounds in each session.
    assert np.all(result["ci_lower"] <= result["estimate"] + 1e-9)
    assert np.all(result["estimate"] <= result["ci_upper"] + 1e-9)


def test_subsample_ci_reproducible_with_seed():
    counts, party, session, X = _dgp_for_ci()
    factory = lambda: LeaveOutEstimator()
    a = subsample_ci(factory, counts, party, session,
                     n_subsamples=10, frac=0.3, seed=42, transform="identity")
    b = subsample_ci(factory, counts, party, session,
                     n_subsamples=10, frac=0.3, seed=42, transform="identity")
    np.testing.assert_allclose(a["subsample_estimates"], b["subsample_estimates"])


def test_subsample_ci_log_transform_returns_valid_interval():
    counts, party, session, X = _dgp_for_ci()
    factory = lambda: LeaveOutEstimator()
    result = subsample_ci(factory, counts, party, session,
                          n_subsamples=20, frac=0.3, seed=1, transform="log")
    assert np.all(result["ci_lower"] >= 0.5 - 1e-6)
    assert np.all(result["ci_upper"] <= 1.0 + 1e-6)


def test_experiment_b_smoke():
    from politext_torch.experiments import mc_coverage
    mc_coverage.N_REP = 3
    mc_coverage.N_SUB = 5
    mc_coverage.run()
