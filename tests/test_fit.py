# tests/test_fit.py
import numpy as np
import torch
from politext_torch._types import PhraseData
from politext_torch.model import PhraseChoiceModel
from politext_torch.fit import fit_mle, fit_penalized, fit_path
from politext_torch.simulate import draw_counts


def _bigger_dgp(V=6, T=2, N=800, P=1, seed=0):
    """Larger DGP for optimization recovery tests."""
    rng = np.random.default_rng(seed)
    alpha = torch.tensor(rng.standard_normal((V, T)) * 0.5, dtype=torch.float32)
    gamma = torch.tensor(rng.standard_normal((V, P)) * 0.2, dtype=torch.float32)
    phi = torch.tensor(rng.standard_normal((V, T)) * 0.5, dtype=torch.float32)
    party = torch.tensor((rng.random(N) < 0.5).astype(np.float32))
    session = torch.tensor(rng.integers(0, T, size=N), dtype=torch.long)
    X = torch.tensor(rng.standard_normal((N, P)).astype(np.float32))
    m = torch.full((N,), 2000.0)
    return dict(alpha=alpha, gamma=gamma, phi=phi,
                party=party, session=session, X=X, m=m,
                V=V, T=T, P=P, N=N)


def _prepare(dgp, seed=1):
    counts = draw_counts(dgp["alpha"], dgp["gamma"], dgp["phi"],
                         dgp["X"], dgp["party"], dgp["session"], dgp["m"], seed=seed)
    return PhraseData.from_arrays(
        counts, dgp["party"].numpy(), dgp["session"].numpy(), dgp["X"].numpy()
    )


def test_fit_mle_recovers_true_params_in_the_large_m_limit():
    dgp = _bigger_dgp()
    data = _prepare(dgp)
    model = PhraseChoiceModel(V=dgp["V"], T=dgp["T"], P=dgp["P"])
    model.init_from_data(data)
    fit_mle(model, data, optimizer="lbfgs", max_iter=100, tol=1e-8)
    # Poisson-MLE of the logit model is identified up to a per-speaker shift;
    # the partisan loading phi_jt is identified (enters as K*phi).
    # Check that phi is close to the truth, relatively:
    err = (model.phi.detach() - dgp["phi"]).abs().mean()
    ref = dgp["phi"].abs().mean()
    assert err / ref < 0.20, f"phi recovery error {err/ref:.3f} too large"


def test_fit_mle_adam_runs_and_reduces_loss():
    dgp = _bigger_dgp(N=300)
    data = _prepare(dgp)
    model = PhraseChoiceModel(V=dgp["V"], T=dgp["T"], P=dgp["P"])
    model.init_from_data(data)
    loss_before = model.poisson_nll(data).item()
    fit_mle(model, data, optimizer="adam", max_iter=500, lr=0.05, tol=1e-7)
    loss_after = model.poisson_nll(data).item()
    assert loss_after < loss_before
