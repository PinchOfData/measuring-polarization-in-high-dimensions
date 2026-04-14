"""sklearn-style partisanship estimators."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from politext_torch._types import PhraseData
from politext_torch.fit import fit_mle, fit_penalized, fit_path
from politext_torch.model import PhraseChoiceModel
from politext_torch.partisanship import (
    leave_out_partisanship,
    partisanship,
)


class BasePartisanshipEstimator:
    """Common API scaffolding for the three estimators."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.partisanship_: np.ndarray | None = None
        self.sessions_: np.ndarray | None = None
        self.vocab_size_: int | None = None
        self.n_covariates_: int | None = None

    def _store_metadata(self, data: PhraseData) -> None:
        self.sessions_ = np.unique(data.session.cpu().numpy())
        self.vocab_size_ = int(data.V)
        self.n_covariates_ = int(data.P)

    def to(self, device: str) -> "BasePartisanshipEstimator":
        self.device = device
        return self


class MLEEstimator(BasePartisanshipEstimator):
    """Plug-in MLE estimator (politext §4.1 / eq. 6)."""

    def __init__(
        self,
        optimizer: str = "lbfgs",
        max_iter: int = 100,
        tol: float = 1e-6,
        ridge: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge

    def fit(
        self,
        counts: sp.csr_matrix,
        party: np.ndarray,
        session: np.ndarray,
        X: np.ndarray | None = None,
        speaker_id: np.ndarray | None = None,
        **fit_kwargs: Any,
    ) -> "MLEEstimator":
        data = PhraseData.from_arrays(counts, party, session, X, device=self.device)
        self._store_metadata(data)
        T = int(data.session.max().item()) + 1
        model = PhraseChoiceModel(V=data.V, T=T, P=data.P).to(self.device)
        model.init_from_data(data)
        fit_mle(
            model, data,
            optimizer=self.optimizer,
            max_iter=self.max_iter, tol=self.tol, ridge=self.ridge,
            **fit_kwargs,
        )
        self.alpha_ = model.alpha.detach().cpu().numpy()
        self.gamma_ = model.gamma.detach().cpu().numpy()
        self.phi_ = model.phi.detach().cpu().numpy()
        pi = partisanship(
            model.alpha.detach().cpu(),
            model.gamma.detach().cpu(),
            model.phi.detach().cpu(),
            data.X.cpu(), data.session.cpu(), data.party.cpu(),
        )
        self.partisanship_ = pi.numpy()
        return self


class LeaveOutEstimator(BasePartisanshipEstimator):
    """Leave-one-speaker-out estimator (politext §4.2 / eq. 8)."""

    def fit(
        self,
        counts: sp.csr_matrix,
        party: np.ndarray,
        session: np.ndarray,
        X: np.ndarray | None = None,
        speaker_id: np.ndarray | None = None,
        **fit_kwargs: Any,
    ) -> "LeaveOutEstimator":
        if X is not None:
            warnings.warn(
                "LeaveOutEstimator does not support covariates; `X` is ignored. "
                "Use PenalizedEstimator if you need covariate-adjusted partisanship.",
                UserWarning, stacklevel=2,
            )
        data = PhraseData.from_arrays(counts, party, session, X=None, device=self.device)
        self._store_metadata(data)
        self.partisanship_ = leave_out_partisanship(
            counts=counts, party=party, session=session, speaker_id=speaker_id,
        )
        return self
