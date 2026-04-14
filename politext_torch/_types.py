"""Typed containers shared across modules."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class PhraseData:
    """Training / scoring inputs prepared for the Poisson NLL."""

    counts_sparse: torch.Tensor     # sparse_coo (N, V)
    log_m: torch.Tensor             # (N,)
    party: torch.Tensor             # (N,) float in {0, 1}
    session: torch.Tensor           # (N,) long
    X: torch.Tensor                 # (N, P)

    @classmethod
    def from_arrays(
        cls,
        counts: sp.csr_matrix,
        party: np.ndarray,
        session: np.ndarray,
        X: np.ndarray | None = None,
        device: str | torch.device = "cpu",
    ) -> "PhraseData":
        coo = counts.tocoo()
        indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
        values = torch.from_numpy(coo.data).float()
        counts_t = torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()
        N = counts.shape[0]
        log_m = torch.log(torch.from_numpy(
            np.asarray(counts.sum(axis=1)).ravel()
        ).float())
        party_t = torch.from_numpy(np.asarray(party, dtype=np.float32))
        session_t = torch.from_numpy(np.asarray(session, dtype=np.int64))
        if X is None:
            X_t = torch.zeros(N, 0)
        else:
            X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        return cls(
            counts_sparse=counts_t.to(device),
            log_m=log_m.to(device),
            party=party_t.to(device),
            session=session_t.to(device),
            X=X_t.to(device),
        )

    @property
    def N(self) -> int:
        return self.counts_sparse.shape[0]

    @property
    def V(self) -> int:
        return self.counts_sparse.shape[1]

    @property
    def P(self) -> int:
        return self.X.shape[1]

    def to(self, device: str | torch.device) -> "PhraseData":
        return PhraseData(
            counts_sparse=self.counts_sparse.to(device),
            log_m=self.log_m.to(device),
            party=self.party.to(device),
            session=self.session.to(device),
            X=self.X.to(device),
        )
