import numpy as np
import pytest
import torch
from politext_torch.estimators import (
    MLEEstimator, LeaveOutEstimator,
)
from politext_torch.simulate import draw_counts


def _big_dgp(V=6, T=2, N=600, P=1, seed=0):
    rng = np.random.default_rng(seed)
    alpha = torch.tensor(rng.standard_normal((V, T)) * 0.5, dtype=torch.float32)
    gamma = torch.tensor(rng.standard_normal((V, P)) * 0.2, dtype=torch.float32)
    phi = torch.tensor(rng.standard_normal((V, T)) * 0.5, dtype=torch.float32)
    party = torch.tensor((rng.random(N) < 0.5).astype(np.float32))
    session = torch.tensor(rng.integers(0, T, size=N), dtype=torch.long)
    X = torch.tensor(rng.standard_normal((N, P)).astype(np.float32))
    m = torch.full((N,), 1500.0)
    counts = draw_counts(alpha, gamma, phi, X, party, session, m, seed=seed + 1)
    return dict(
        counts=counts, party=party.numpy(), session=session.numpy(),
        X=X.numpy(), V=V, T=T, P=P, N=N,
    )


def test_mle_estimator_fit_populates_attrs():
    d = _big_dgp(N=300)
    est = MLEEstimator(max_iter=50)
    est.fit(d["counts"], d["party"], d["session"], d["X"])
    assert est.partisanship_.shape == (d["T"],)
    assert est.alpha_.shape == (d["V"], d["T"])
    assert est.gamma_.shape == (d["V"], d["P"])
    assert est.phi_.shape == (d["V"], d["T"])
    # Partisanship must be in [0, 1]
    assert np.all((est.partisanship_ >= 0) & (est.partisanship_ <= 1))


def test_leaveout_estimator_ignores_X_with_warning():
    d = _big_dgp(N=200)
    est = LeaveOutEstimator()
    with pytest.warns(UserWarning, match="ignored"):
        est.fit(d["counts"], d["party"], d["session"], X=d["X"])
    assert est.partisanship_.shape == (d["T"],)
