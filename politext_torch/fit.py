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


def _soft_threshold(x: torch.Tensor, thresh: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(x.abs() - thresh, min=0.0)


def _smooth_part(
    model: PhraseChoiceModel,
    data: PhraseData,
    batch_size: int,
    ridge_alpha: float,
    ridge_gamma: float,
) -> torch.Tensor:
    return model.poisson_nll(
        data, batch_size=batch_size,
        ridge_alpha=ridge_alpha, ridge_gamma=ridge_gamma,
    )


def fit_penalized(
    model: PhraseChoiceModel,
    data: PhraseData,
    lam: float,
    lam_alpha: float = 1e-5,
    lam_gamma: float = 1e-5,
    max_iter: int = 500,
    tol: float = 1e-5,
    backtracking: bool = True,
    batch_size: int = 512,
    verbose: bool = False,
) -> dict:
    """FISTA: Poisson NLL + 0.5*lam_alpha*||alpha||^2 + 0.5*lam_gamma*||gamma||^2
            + lam*||phi||_1.

    L2 terms are inside the smooth part; L1 on phi is handled by prox step.
    """
    history = {"loss": [], "step": []}

    # Initial extrapolation point y = current theta.
    def snapshot():
        return (model.alpha.detach().clone(),
                model.gamma.detach().clone(),
                model.phi.detach().clone())

    theta_prev = snapshot()
    theta_y = snapshot()
    t_k = 1.0
    eta = 1.0  # initial step size

    def objective(a, g, p):
        # Full objective including L1. For diagnostics only.
        with torch.no_grad():
            orig = snapshot()
            model.alpha.copy_(a); model.gamma.copy_(g); model.phi.copy_(p)
            smooth = _smooth_part(model, data, batch_size, lam_alpha, lam_gamma).item()
            model.alpha.copy_(orig[0]); model.gamma.copy_(orig[1]); model.phi.copy_(orig[2])
        return smooth + lam * p.abs().sum().item()

    prev_obj = None

    for it in range(max_iter):
        # Gradient at y
        with torch.enable_grad():
            model.alpha.data.copy_(theta_y[0])
            model.gamma.data.copy_(theta_y[1])
            model.phi.data.copy_(theta_y[2])
            model.alpha.grad = None
            model.gamma.grad = None
            model.phi.grad = None
            smooth_y = _smooth_part(model, data, batch_size, lam_alpha, lam_gamma)
            smooth_y.backward()
            g_alpha = model.alpha.grad.detach().clone()
            g_gamma = model.gamma.grad.detach().clone()
            g_phi = model.phi.grad.detach().clone()

        f_y = smooth_y.detach().item()
        a_y, g_y, p_y = theta_y

        # Backtracking line search
        while True:
            alpha_new = a_y - eta * g_alpha
            gamma_new = g_y - eta * g_gamma
            phi_new = _soft_threshold(p_y - eta * g_phi, eta * lam)

            with torch.no_grad():
                model.alpha.copy_(alpha_new)
                model.gamma.copy_(gamma_new)
                model.phi.copy_(phi_new)
                f_new = _smooth_part(model, data, batch_size, lam_alpha, lam_gamma).item()

            # Majorization Q(theta_new; y) = f(y) + <grad, theta_new - y>
            #   + (1 / (2 eta)) ||theta_new - y||^2
            d_alpha = alpha_new - a_y
            d_gamma = gamma_new - g_y
            d_phi = phi_new - p_y
            quad = (d_alpha.pow(2).sum() + d_gamma.pow(2).sum()
                    + d_phi.pow(2).sum()).item()
            linear = ((g_alpha * d_alpha).sum()
                      + (g_gamma * d_gamma).sum()
                      + (g_phi * d_phi).sum()).item()
            Q = f_y + linear + quad / (2 * eta)

            if not backtracking or f_new <= Q + 1e-10:
                break
            eta *= 0.5
            if eta < 1e-12:
                break

        # Accept step
        theta_new = (alpha_new.detach().clone(),
                     gamma_new.detach().clone(),
                     phi_new.detach().clone())
        cur_obj = f_new + lam * phi_new.abs().sum().item()
        history["loss"].append(cur_obj)
        history["step"].append(eta)

        if prev_obj is not None and abs(prev_obj - cur_obj) / max(abs(prev_obj), 1e-8) < tol:
            break
        prev_obj = cur_obj

        # Nesterov extrapolation
        t_next = (1 + (1 + 4 * t_k ** 2) ** 0.5) / 2
        beta = (t_k - 1) / t_next
        theta_y = (
            theta_new[0] + beta * (theta_new[0] - theta_prev[0]),
            theta_new[1] + beta * (theta_new[1] - theta_prev[1]),
            theta_new[2] + beta * (theta_new[2] - theta_prev[2]),
        )
        t_k = t_next
        theta_prev = theta_new

        # Gentle step-size warm-up across iterations (prevents eta from only shrinking)
        eta = min(eta * 1.1, 1e3)

        if verbose and it % 50 == 0:
            print(f"iter {it}: obj={cur_obj:.4f}  eta={eta:.3g}")

    # Write back the last accepted iterate
    with torch.no_grad():
        model.alpha.copy_(theta_new[0])
        model.gamma.copy_(theta_new[1])
        model.phi.copy_(theta_new[2])

    return history


def _compute_lambda_max(
    model: PhraseChoiceModel,
    data: PhraseData,
    batch_size: int,
) -> float:
    """Smallest lambda that keeps phi = 0. Equals max_{j,t} |dNLL/dphi_{j,t}| at phi=0."""
    with torch.no_grad():
        orig_phi = model.phi.detach().clone()
        model.phi.zero_()
    model.phi.requires_grad_(True)
    if model.phi.grad is not None:
        model.phi.grad.zero_()
    loss = model.poisson_nll(data, batch_size=batch_size)
    grad = torch.autograd.grad(loss, model.phi)[0]
    with torch.no_grad():
        model.phi.copy_(orig_phi)
    return grad.detach().abs().max().item()


def _compute_bic(log_lik: float, df: int, n: int) -> float:
    import math
    return -2.0 * log_lik + math.log(max(n, 1)) * df


def fit_path(
    model: PhraseChoiceModel,
    data: PhraseData,
    lam_grid: list[float] | None = None,
    grid_size: int = 100,
    lam_min_ratio: float = 1e-3,
    criterion: str = "bic",
    lam_alpha: float = 1e-5,
    lam_gamma: float = 1e-5,
    max_iter: int = 500,
    tol: float = 1e-5,
    batch_size: int = 512,
    store_path_params: bool = False,
    verbose: bool = False,
) -> dict:
    """L1 regularization path with warm starts. Returns selected lambda and diagnostics.

    Grid order: DECREASING lambda (coarse to fine).
    """
    if lam_grid is None:
        lam_max = _compute_lambda_max(model, data, batch_size=batch_size)
        import numpy as np
        lam_grid = np.geomspace(lam_max, lam_max * lam_min_ratio, grid_size).tolist()
    else:
        lam_grid = list(lam_grid)

    path = []
    stored = [] if store_path_params else None
    for lam in lam_grid:
        fit_penalized(
            model, data, lam=lam,
            lam_alpha=lam_alpha, lam_gamma=lam_gamma,
            max_iter=max_iter, tol=tol,
            batch_size=batch_size, verbose=False,
        )
        with torch.no_grad():
            log_lik = -model.poisson_nll(data, batch_size=batch_size).item()
            df = int((model.phi.detach().abs() > 1e-8).sum().item())
            bic = _compute_bic(log_lik, df, n=data.N)
        path.append({"lam": lam, "logLik": log_lik, "df": df, "bic": bic})
        if stored is not None:
            stored.append((
                model.alpha.detach().clone(),
                model.gamma.detach().clone(),
                model.phi.detach().clone(),
            ))
        if verbose:
            print(f"lam={lam:.4g}  logLik={log_lik:.3f}  df={df}  bic={bic:.3f}")

    if criterion == "bic":
        best_idx = min(range(len(path)), key=lambda i: path[i]["bic"])
    else:
        raise NotImplementedError(f"criterion={criterion!r} not yet supported")

    # Restore model to best iterate
    if stored is not None:
        with torch.no_grad():
            a, g, p = stored[best_idx]
            model.alpha.copy_(a); model.gamma.copy_(g); model.phi.copy_(p)
    else:
        # Re-fit at the best lambda for a clean final iterate
        best_lam = path[best_idx]["lam"]
        fit_penalized(
            model, data, lam=best_lam,
            lam_alpha=lam_alpha, lam_gamma=lam_gamma,
            max_iter=max_iter, tol=tol,
            batch_size=batch_size,
        )

    return {
        "lam": path[best_idx]["lam"],
        "best_idx": best_idx,
        "path": path,
        "path_params": stored,
    }
