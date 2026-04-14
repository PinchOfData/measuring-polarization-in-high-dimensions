import numpy as np
import torch
from politext_torch.partisanship import (
    choice_probs, posterior_rho, partisanship,
    leave_out_partisanship,
)


def test_choice_probs_equals_softmax(tiny_dgp):
    d = tiny_dgp
    t = 0
    X_row = d["X"][0]
    q_R = choice_probs(d["alpha"][:, t], d["gamma"], d["phi"][:, t],
                       X_row=X_row, party=1.0)
    u = d["alpha"][:, t] + X_row @ d["gamma"].T + d["phi"][:, t] * 1.0
    torch.testing.assert_close(q_R, torch.softmax(u, dim=-1))


def test_posterior_rho_sums_to_one_with_complement(tiny_dgp):
    d = tiny_dgp
    t = 0
    rho = posterior_rho(d["alpha"][:, t], d["gamma"], d["phi"][:, t], X_row=d["X"][0])
    assert (rho >= 0).all() and (rho <= 1).all()


def test_partisanship_hand_computed():
    """2-phrase, 2-speaker (1R, 1D), 1-session, 1-covariate example."""
    V, T, P = 2, 1, 1
    alpha = torch.tensor([[0.0], [0.0]])
    gamma = torch.zeros(V, P)
    phi = torch.tensor([[1.0], [-1.0]])
    X = torch.tensor([[0.0], [0.0]])
    party = torch.tensor([1.0, 0.0])
    session = torch.tensor([0, 0], dtype=torch.long)

    # For R (party=1): u = (1, -1); q^R = softmax = (e/(e+1/e), ...)
    # For D (party=0): u = (0, 0);  q^D = (0.5, 0.5)
    q_R = torch.tensor([np.exp(1), np.exp(-1)])
    q_R = q_R / q_R.sum()
    q_D = torch.tensor([0.5, 0.5])
    rho = q_R / (q_R + q_D)
    # pi_t(x) = 0.5 * q^R * rho + 0.5 * q^D * (1 - rho)
    pi_R = 0.5 * (q_R * rho).sum() + 0.5 * (q_D * (1 - rho)).sum()
    pi_D = pi_R  # both speakers have the same x
    expected = 0.5 * (pi_R + pi_D)

    pi = partisanship(alpha, gamma, phi, X, session, party)
    assert pi.shape == (1,)
    torch.testing.assert_close(pi, expected.reshape(1), atol=1e-5, rtol=1e-5)


def test_leave_out_partisanship_reference_match():
    """Compare to a slow Python implementation on a small example."""
    rng = np.random.default_rng(3)
    V, T, N = 5, 2, 12
    session = np.array([t for t in range(T) for _ in range(N // T)])
    party = rng.integers(0, 2, size=N).astype(float)
    m = np.full(N, 100)
    import scipy.sparse as sp
    counts_dense = rng.multinomial(100, [1/V]*V, size=N)
    counts = sp.csr_matrix(counts_dense)

    # Slow reference
    def ref():
        pi = np.zeros(T)
        for t in range(T):
            idx_t = np.where(session == t)[0]
            idx_R = [i for i in idx_t if party[i] == 1]
            idx_D = [i for i in idx_t if party[i] == 0]
            if not idx_R or not idx_D:
                pi[t] = np.nan
                continue
            sum_R = counts_dense[idx_R].sum(axis=0)
            sum_D = counts_dense[idx_D].sum(axis=0)
            m_R = counts_dense[idx_R].sum()
            m_D = counts_dense[idx_D].sum()
            total_R = 0.0
            for i in idx_R:
                q_i = counts_dense[i] / counts_dense[i].sum()
                exc_R = sum_R - counts_dense[i]
                exc_m_R = m_R - counts_dense[i].sum()
                qhat_R = exc_R / exc_m_R
                qhat_D = sum_D / m_D
                rho = np.where(qhat_R + qhat_D > 0,
                               qhat_R / (qhat_R + qhat_D + 1e-30),
                               0.5)
                total_R += (q_i * rho).sum()
            total_D = 0.0
            for i in idx_D:
                q_i = counts_dense[i] / counts_dense[i].sum()
                exc_D = sum_D - counts_dense[i]
                exc_m_D = m_D - counts_dense[i].sum()
                qhat_D = exc_D / exc_m_D
                qhat_R = sum_R / m_R
                rho = np.where(qhat_R + qhat_D > 0,
                               qhat_R / (qhat_R + qhat_D + 1e-30),
                               0.5)
                total_D += (q_i * (1 - rho)).sum()
            pi[t] = 0.5 * total_R / len(idx_R) + 0.5 * total_D / len(idx_D)
        return pi

    expected = ref()
    got = leave_out_partisanship(
        counts=counts,
        party=party, session=session,
        speaker_id=np.arange(N),
    )
    np.testing.assert_allclose(got, expected, atol=1e-6)
