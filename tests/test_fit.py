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

    def _center_phrases(x):
        # phi has shape (V, T); mean over phrases within session
        return x - x.mean(dim=0, keepdim=True)

    # The Poisson plug-in MLE (paper eq. 9, spec §5.1) differs from the
    # multinomial DGP truth by per-session shifts c_t that are absorbed into
    # alpha for R-speakers; these shifts are not identified from the Poisson
    # likelihood alone. Only contrasts of phi across phrases within a session
    # are identified, so we compare phi after centering each column (session).
    phi_hat_c = _center_phrases(model.phi.detach())
    phi_true_c = _center_phrases(dgp["phi"])
    err = (phi_hat_c - phi_true_c).abs().mean()
    ref = phi_true_c.abs().mean()
    assert err / ref < 0.10, (
        f"centered-phi recovery error {err/ref:.3f} too large "
        f"(per-session shifts are absorbed by alpha; we only compare identified contrasts)"
    )


def test_fit_mle_adam_runs_and_reduces_loss():
    dgp = _bigger_dgp(N=300)
    data = _prepare(dgp)
    model = PhraseChoiceModel(V=dgp["V"], T=dgp["T"], P=dgp["P"])
    model.init_from_data(data)
    loss_before = model.poisson_nll(data).item()
    fit_mle(model, data, optimizer="adam", max_iter=500, lr=0.05, tol=1e-7)
    loss_after = model.poisson_nll(data).item()
    assert loss_after < loss_before


def test_fit_penalized_large_lambda_zeros_phi():
    dgp = _bigger_dgp(N=300)
    data = _prepare(dgp)
    model = PhraseChoiceModel(V=dgp["V"], T=dgp["T"], P=dgp["P"])
    model.init_from_data(data)
    # Very large lambda should drive phi to exactly 0 after soft-thresholding.
    fit_penalized(model, data, lam=1e6, max_iter=200, tol=1e-6)
    assert (model.phi.detach().abs() < 1e-8).all()


def test_fit_penalized_small_lambda_close_to_mle():
    dgp = _bigger_dgp(N=600)
    data = _prepare(dgp)
    # MLE reference
    m_mle = PhraseChoiceModel(V=dgp["V"], T=dgp["T"], P=dgp["P"])
    m_mle.init_from_data(data)
    fit_mle(m_mle, data, max_iter=100, tol=1e-8)
    # Penalized with tiny lambda
    m_pen = PhraseChoiceModel(V=dgp["V"], T=dgp["T"], P=dgp["P"])
    m_pen.init_from_data(data)
    fit_penalized(m_pen, data, lam=1e-4, max_iter=2000, tol=1e-8)
    err = (m_pen.phi.detach() - m_mle.phi.detach()).abs().mean()
    assert err < 0.05, f"penalized with tiny lambda should match MLE, got {err:.3f}"
