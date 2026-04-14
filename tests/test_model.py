import numpy as np
import pytest
import scipy.sparse as sp
import torch
from politext_torch.model import PhraseChoiceModel, PhraseData


def make_tiny_data(tiny_dgp, seed=0):
    from politext_torch.simulate import draw_counts
    d = tiny_dgp
    counts = draw_counts(d["alpha"], d["gamma"], d["phi"], d["X"],
                         d["party"], d["session"], d["m"], seed=seed)
    return PhraseData.from_arrays(
        counts=counts, party=d["party"].numpy(),
        session=d["session"].numpy(), X=d["X"].numpy(),
    )


def test_model_param_shapes(tiny_dgp):
    model = PhraseChoiceModel(V=tiny_dgp["V"], T=tiny_dgp["T"], P=tiny_dgp["P"])
    assert model.alpha.shape == (4, 2)
    assert model.gamma.shape == (4, 1)
    assert model.phi.shape == (4, 2)


def test_model_init_from_data_uses_empirical_log_freq(tiny_dgp):
    data = make_tiny_data(tiny_dgp)
    model = PhraseChoiceModel(V=tiny_dgp["V"], T=tiny_dgp["T"], P=tiny_dgp["P"])
    model.init_from_data(data)
    # phi and gamma start at zero
    torch.testing.assert_close(model.phi, torch.zeros_like(model.phi))
    torch.testing.assert_close(model.gamma, torch.zeros_like(model.gamma))
    # alpha[:, t] should be log((sum c in t + eps) / (sum m in t + eps))
    totals_per_session = torch.zeros(tiny_dgp["T"], tiny_dgp["V"])
    m_per_session = torch.zeros(tiny_dgp["T"])
    counts = data.counts_sparse.to_dense()
    for i in range(tiny_dgp["N"]):
        t = tiny_dgp["session"][i]
        totals_per_session[t] += counts[i]
        m_per_session[t] += data.log_m[i].exp()
    eps = 1e-6
    expected = torch.log((totals_per_session + eps) / (m_per_session[:, None] + eps))
    torch.testing.assert_close(model.alpha, expected.T, atol=1e-5, rtol=1e-5)


def _reference_poisson_nll(model, data):
    """Slow numpy reference: exactly sums over every (i, j)."""
    alpha_np = model.alpha.detach().numpy()
    gamma_np = model.gamma.detach().numpy()
    phi_np = model.phi.detach().numpy()
    counts = data.counts_sparse.to_dense().numpy()
    log_m = data.log_m.numpy()
    party = data.party.numpy()
    session = data.session.numpy()
    X = data.X.numpy()

    rate_sum = 0.0
    data_fit = 0.0
    N, V = counts.shape
    for i in range(N):
        t = session[i]
        K = party[i]
        u_i = alpha_np[:, t] + X[i] @ gamma_np.T + phi_np[:, t] * K
        rates = np.exp(log_m[i] + u_i)
        rate_sum += rates.sum()
        data_fit += (counts[i] * u_i).sum()
    return rate_sum - data_fit


def test_poisson_nll_matches_numpy_reference(tiny_dgp):
    import numpy as np
    data = make_tiny_data(tiny_dgp)
    model = PhraseChoiceModel(V=tiny_dgp["V"], T=tiny_dgp["T"], P=tiny_dgp["P"])
    # use a non-trivial param value so we're testing the formula, not zeros
    with torch.no_grad():
        model.alpha.copy_(tiny_dgp["alpha"])
        model.gamma.copy_(tiny_dgp["gamma"])
        model.phi.copy_(tiny_dgp["phi"])
    nll_t = model.poisson_nll(data)
    nll_ref = _reference_poisson_nll(model, data)
    np.testing.assert_allclose(nll_t.item(), nll_ref, rtol=1e-5)


def test_poisson_nll_batch_size_invariant(tiny_dgp):
    data = make_tiny_data(tiny_dgp)
    model = PhraseChoiceModel(V=tiny_dgp["V"], T=tiny_dgp["T"], P=tiny_dgp["P"])
    with torch.no_grad():
        model.alpha.copy_(tiny_dgp["alpha"])
        model.phi.copy_(tiny_dgp["phi"])
    # Equal within float tolerance regardless of batch_size.
    nll_1 = model.poisson_nll(data, batch_size=1).item()
    nll_big = model.poisson_nll(data, batch_size=1024).item()
    assert abs(nll_1 - nll_big) < 1e-4


def test_poisson_nll_is_differentiable(tiny_dgp):
    data = make_tiny_data(tiny_dgp)
    model = PhraseChoiceModel(V=tiny_dgp["V"], T=tiny_dgp["T"], P=tiny_dgp["P"])
    nll = model.poisson_nll(data)
    nll.backward()
    # all three params must have gradients
    assert model.alpha.grad is not None
    assert model.gamma.grad is not None
    assert model.phi.grad is not None
    assert torch.isfinite(model.alpha.grad).all()
