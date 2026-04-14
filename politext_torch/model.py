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
