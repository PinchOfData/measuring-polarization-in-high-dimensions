"""PhraseChoiceModel: parameters and Poisson NLL (Gentzkow-Shapiro-Taddy 2019)."""
from __future__ import annotations

import torch
import torch.nn as nn

from politext_torch._types import PhraseData

__all__ = ["PhraseChoiceModel", "PhraseData"]


class PhraseChoiceModel(nn.Module):
    """Poisson-approximated multinomial-logit phrase-choice model.

    Parameters
    ----------
    V : vocabulary size.
    T : number of sessions.
    P : number of covariate columns.
    """

    def __init__(self, V: int, T: int, P: int):
        super().__init__()
        self.V, self.T, self.P = V, T, P
        self.alpha = nn.Parameter(torch.zeros(V, T))
        self.gamma = nn.Parameter(torch.zeros(V, P))
        self.phi = nn.Parameter(torch.zeros(V, T))

    @torch.no_grad()
    def init_from_data(self, data: PhraseData, eps: float = 1e-6) -> None:
        """Initialize alpha to empirical log-frequencies per session; gamma, phi to 0."""
        T, V = self.T, self.V
        counts_dense = data.counts_sparse.to_dense()
        m = data.log_m.exp()
        alpha_new = torch.zeros(V, T, device=self.alpha.device)
        for t in range(T):
            mask = data.session == t
            if not mask.any():
                continue
            phrase_totals = counts_dense[mask].sum(dim=0)
            m_total = m[mask].sum()
            alpha_new[:, t] = torch.log((phrase_totals + eps) / (m_total + eps))
        self.alpha.copy_(alpha_new)
        self.gamma.zero_()
        self.phi.zero_()

    def poisson_nll(
        self,
        data: PhraseData,
        batch_size: int = 512,
        ridge_alpha: float = 0.0,
        ridge_gamma: float = 0.0,
    ) -> torch.Tensor:
        """Poisson NLL (negated log-likelihood, to minimize).

        Rate sum is accumulated in document-batches of `batch_size`.
        Data-fit term is computed on the sparse nnz entries only.
        """
        N, V = data.N, data.V
        session = data.session
        party = data.party
        X = data.X
        log_m = data.log_m

        rate_sum = X.new_zeros(())
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            sl = slice(start, end)
            u_B = (
                self.alpha[:, session[sl]].T                     # (|B|, V)
                + X[sl] @ self.gamma.T                           # (|B|, V)
                + self.phi[:, session[sl]].T * party[sl, None]   # (|B|, V)
            )
            rate_sum = rate_sum + torch.exp(log_m[sl, None] + u_B).sum()

        coo = data.counts_sparse.coalesce()
        idx = coo.indices()         # (2, nnz)
        vals = coo.values()         # (nnz,)
        i_idx, j_idx = idx[0], idx[1]
        u_obs = (
            self.alpha[j_idx, session[i_idx]]
            + (X[i_idx] * self.gamma[j_idx]).sum(dim=-1)
            + self.phi[j_idx, session[i_idx]] * party[i_idx]
        )
        data_fit = (vals * u_obs).sum()

        loss = rate_sum - data_fit
        if ridge_alpha > 0:
            loss = loss + 0.5 * ridge_alpha * (self.alpha ** 2).sum()
        if ridge_gamma > 0:
            loss = loss + 0.5 * ridge_gamma * (self.gamma ** 2).sum()
        return loss
