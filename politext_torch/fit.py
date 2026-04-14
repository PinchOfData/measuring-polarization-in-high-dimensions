# politext_torch/fit.py
"""Fitters for PhraseChoiceModel: fit_mle, fit_penalized, fit_path."""
from __future__ import annotations

from typing import Callable

import torch

from politext_torch._types import PhraseData
from politext_torch.model import PhraseChoiceModel


def fit_mle(
    model: PhraseChoiceModel,
    data: PhraseData,
    optimizer: str = "lbfgs",
    max_iter: int = 100,
    tol: float = 1e-6,
    ridge: float = 0.0,
    lr: float = 0.05,
    batch_size: int = 512,
    verbose: bool = False,
) -> dict:
    """Unpenalized Poisson-NLL fit. Convex in all params.

    optimizer : 'lbfgs' (default) or 'adam'.
    """
    history = {"loss": []}
    params = [model.alpha, model.gamma, model.phi]

    if optimizer == "lbfgs":
        opt = torch.optim.LBFGS(
            params, lr=lr, max_iter=max_iter, tolerance_change=tol,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            loss = model.poisson_nll(
                data, batch_size=batch_size,
                ridge_alpha=ridge, ridge_gamma=ridge,
            )
            loss.backward()
            history["loss"].append(loss.item())
            return loss

        opt.step(closure)
        return history

    if optimizer == "adam":
        opt = torch.optim.Adam(params, lr=lr)
        prev = None
        for it in range(max_iter):
            opt.zero_grad()
            loss = model.poisson_nll(
                data, batch_size=batch_size,
                ridge_alpha=ridge, ridge_gamma=ridge,
            )
            loss.backward()
            opt.step()
            cur = loss.item()
            history["loss"].append(cur)
            if verbose and it % 50 == 0:
                print(f"iter {it}: loss={cur:.4f}")
            if prev is not None and abs(prev - cur) / max(abs(prev), 1e-8) < tol:
                break
            prev = cur
        return history

    raise ValueError(f"Unknown optimizer: {optimizer!r}")


def fit_penalized(*args, **kwargs):
    """Placeholder — implemented in Task 6."""
    raise NotImplementedError("fit_penalized is implemented in Task 6")


def fit_path(*args, **kwargs):
    """Placeholder — implemented in Task 7."""
    raise NotImplementedError("fit_path is implemented in Task 7")
