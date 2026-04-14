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
