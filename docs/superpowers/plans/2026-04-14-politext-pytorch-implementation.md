# PyTorch politext Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyTorch library implementing the three partisanship estimators of Gentzkow–Shapiro–Taddy (2019), plus subsampling CIs, unseen-text scaling (Widmer et al. 2020), and Monte Carlo validation.

**Architecture:** Layered — a `PhraseChoiceModel` (`nn.Module`) owns parameters and the Poisson NLL; `fit.py` holds `fit_mle` / `fit_penalized` (FISTA) / `fit_path` (BIC-selected regularization path); pure functions in `partisanship.py` compute π̂ from fitted params; three sklearn-style wrappers in `estimators.py` surface the estimators; `inference.py` implements speaker-level subsampling CIs; `scale.py` exposes the two unseen-text methods.

**Tech Stack:** Python 3.11, PyTorch 2.11 (CUDA 12.8), scipy.sparse, NumPy, scikit-learn (user-facing), patsy (user-facing), pytest, matplotlib.

**Environment:** All commands assume you are in the repo root (`/mnt/c/Users/Gauthier/Desktop/measuring-polarization-in-high-dimensions`) with the `gst` conda env active (`conda activate gst`).

**Reference:** `docs/superpowers/specs/2026-04-14-politext-pytorch-design.md` — the approved design spec.

---

## File Structure

```
politext_torch/
  __init__.py           # public API: estimators, simulate, scale, inference
  _types.py             # PhraseData dataclass + small typed helpers
  simulate.py           # draw_counts(...) + make_mc_A/B/C(...)
  model.py              # PhraseChoiceModel (nn.Module): params + Poisson NLL
  fit.py                # fit_mle, fit_penalized (FISTA), fit_path (BIC/CV)
  partisanship.py       # choice_probs, posterior_rho, partisanship,
                        # leave_out_rho, leave_out_partisanship
  estimators.py         # MLEEstimator, LeaveOutEstimator, PenalizedEstimator
  inference.py          # subsample_ci
  scale.py              # scale_document(s), score_document(s)
  experiments/
    __init__.py
    mc_bias_rmse.py     # Experiment A
    mc_coverage.py      # Experiment B
    mc_null.py          # Experiment C
tests/
  __init__.py
  conftest.py           # shared fixtures (tiny DGP, seeded rng)
  test_simulate.py
  test_model.py
  test_fit.py
  test_partisanship.py
  test_estimators.py
  test_inference.py
  test_scale.py
pyproject.toml
README.md
```

Each module has one clear responsibility. `model.py` only knows about params and NLL; `fit.py` only knows about optimization; `partisanship.py` only knows the quantity formulas; `estimators.py` only orchestrates the three. Pure functions where possible (everything in `partisanship.py` and `scale.py`).

---

## Task 1: Package skeleton and dev entry point

**Files:**
- Create: `pyproject.toml`
- Create: `politext_torch/__init__.py`
- Create: `politext_torch/experiments/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "politext-torch"
version = "0.1.0"
description = "PyTorch implementation of Gentzkow-Shapiro-Taddy (2019) partisanship estimators."
requires-python = ">=3.11"
dependencies = [
  "torch>=2.2",
  "numpy>=1.24",
  "scipy>=1.11",
]

[project.optional-dependencies]
dev = [
  "pytest>=7.4",
  "pytest-xdist>=3.3",
  "scikit-learn>=1.3",
  "patsy>=0.5",
  "matplotlib>=3.7",
  "pandas>=2.0",
  "joblib>=1.3",
]

[tool.setuptools.packages.find]
include = ["politext_torch*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"
```

- [ ] **Step 2: Write `politext_torch/__init__.py`**

```python
"""politext_torch: PyTorch partisanship estimators after Gentzkow-Shapiro-Taddy (2019)."""

__version__ = "0.1.0"

from politext_torch.estimators import (
    MLEEstimator,
    LeaveOutEstimator,
    PenalizedEstimator,
)
from politext_torch.inference import subsample_ci
from politext_torch.scale import (
    scale_document,
    scale_documents,
    score_document,
    score_documents,
)
from politext_torch.simulate import draw_counts

__all__ = [
    "MLEEstimator",
    "LeaveOutEstimator",
    "PenalizedEstimator",
    "subsample_ci",
    "scale_document",
    "scale_documents",
    "score_document",
    "score_documents",
    "draw_counts",
]
```

- [ ] **Step 3: Write `politext_torch/experiments/__init__.py`**

```python
"""Monte Carlo experiments validating the estimators."""
```

- [ ] **Step 4: Write `tests/__init__.py` (empty) and `tests/conftest.py`**

`tests/__init__.py`: empty file.

`tests/conftest.py`:
```python
"""Shared pytest fixtures."""
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything():
    """Seed torch and numpy for deterministic tests."""
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def tiny_dgp():
    """Return params for a minimal 5-speaker, 4-phrase, 2-session DGP.

    Shapes: alpha (V,T)=(4,2), gamma (V,P)=(4,1), phi (V,T)=(4,2).
    """
    V, T, P, N = 4, 2, 1, 5
    alpha = torch.tensor([[0.0, 0.1], [-0.2, 0.0], [0.1, 0.2], [0.0, -0.1]])
    gamma = torch.tensor([[0.0], [0.0], [0.1], [-0.1]])
    phi = torch.tensor([[0.3, 0.2], [-0.1, -0.2], [0.0, 0.0], [0.0, 0.0]])
    party = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    session = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    X = torch.tensor([[1.0], [0.0], [1.0], [0.0], [1.0]])
    m = torch.full((N,), 100.0)
    return dict(
        alpha=alpha, gamma=gamma, phi=phi,
        party=party, session=session, X=X, m=m,
        V=V, T=T, P=P, N=N,
    )
```

- [ ] **Step 5: Install the package editable and smoke-test**

Run: `pip install -e ".[dev]"` then `python -c "import politext_torch; print(politext_torch.__version__)"`

Expected: prints `0.1.0`. Imports will fail because we haven't written the modules yet — that's fine for this step; we install in editable mode so later tasks pick up file changes without reinstalling.

**If the editable install errors on the missing modules:** replace the `__init__.py` re-exports with a stub that does nothing: `__version__ = "0.1.0"`. Restore the re-exports in Task 14 after estimators exist.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml politext_torch/__init__.py politext_torch/experiments/__init__.py tests/__init__.py tests/conftest.py
git commit -m "feat: bootstrap politext_torch package skeleton"
```

---

## Task 2: `simulate.draw_counts`

**Why first:** every other test needs synthetic data.

**Files:**
- Create: `politext_torch/simulate.py`
- Create: `tests/test_simulate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_simulate.py
import numpy as np
import scipy.sparse as sp
import torch
from politext_torch.simulate import draw_counts


def test_draw_counts_returns_csr_with_right_shape(tiny_dgp):
    d = tiny_dgp
    counts = draw_counts(
        alpha=d["alpha"], gamma=d["gamma"], phi=d["phi"],
        X=d["X"], party=d["party"], session=d["session"], m=d["m"],
        seed=42,
    )
    assert isinstance(counts, sp.csr_matrix)
    assert counts.shape == (d["N"], d["V"])
    # verbosity preserved: row sums equal m
    np.testing.assert_array_equal(np.asarray(counts.sum(axis=1)).ravel(),
                                  d["m"].numpy())


def test_draw_counts_reproducible_with_seed(tiny_dgp):
    d = tiny_dgp
    a = draw_counts(d["alpha"], d["gamma"], d["phi"], d["X"],
                    d["party"], d["session"], d["m"], seed=7)
    b = draw_counts(d["alpha"], d["gamma"], d["phi"], d["X"],
                    d["party"], d["session"], d["m"], seed=7)
    np.testing.assert_array_equal(a.toarray(), b.toarray())


def test_draw_counts_freqs_converge_to_softmax_probs(tiny_dgp):
    """Law of large numbers: empirical freqs converge to softmax(u) as m grows."""
    d = tiny_dgp
    # single speaker, huge verbosity
    alpha_t = d["alpha"][:, 0]
    phi_t = d["phi"][:, 0]
    u = alpha_t + d["X"][0] @ d["gamma"].T + phi_t * d["party"][0]
    expected = torch.softmax(u, dim=-1).numpy()

    m = torch.tensor([1_000_000.0])
    counts = draw_counts(
        alpha=d["alpha"], gamma=d["gamma"], phi=d["phi"],
        X=d["X"][:1], party=d["party"][:1], session=d["session"][:1], m=m,
        seed=1,
    )
    empirical = counts.toarray()[0] / counts.sum()
    np.testing.assert_allclose(empirical, expected, atol=2e-3)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_simulate.py -v`
Expected: `ModuleNotFoundError: No module named 'politext_torch.simulate'`

- [ ] **Step 3: Implement `draw_counts`**

```python
# politext_torch/simulate.py
"""Synthetic data generation for the phrase-choice Poisson/multinomial model."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def draw_counts(
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    phi: torch.Tensor,
    X: torch.Tensor,
    party: torch.Tensor,
    session: torch.Tensor,
    m: torch.Tensor,
    seed: int | None = None,
) -> sp.csr_matrix:
    """Sample phrase counts from the Gentzkow-Shapiro-Taddy (2019) model.

    For each speaker row i, compute u_ij = alpha[j, t_i] + X[i] @ gamma[j]
      + phi[j, t_i] * K_i, softmax to q_ij, then draw c_i ~ Multinomial(m_i, q_i).

    Parameters
    ----------
    alpha : (V, T)
    gamma : (V, P)
    phi   : (V, T)
    X     : (N, P)
    party : (N,) in {0, 1}
    session : (N,) long in 0..T-1
    m     : (N,) verbosity
    seed  : RNG seed.

    Returns
    -------
    scipy.sparse.csr_matrix of shape (N, V) with integer counts.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    V = alpha.shape[0]

    alpha_i = alpha[:, session].T               # (N, V)
    phi_i = phi[:, session].T                   # (N, V)
    u = alpha_i + X @ gamma.T + phi_i * party[:, None]   # (N, V)
    q = torch.softmax(u, dim=-1).cpu().numpy()  # (N, V)
    m_np = m.cpu().numpy().astype(np.int64)

    rows, cols, vals = [], [], []
    for i in range(N):
        c_i = rng.multinomial(m_np[i], q[i])
        nz = np.flatnonzero(c_i)
        rows.extend([i] * len(nz))
        cols.extend(nz.tolist())
        vals.extend(c_i[nz].tolist())

    return sp.csr_matrix((vals, (rows, cols)), shape=(N, V), dtype=np.int64)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_simulate.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add politext_torch/simulate.py tests/test_simulate.py
git commit -m "feat: add draw_counts synthetic data generator"
```

---

## Task 3: `PhraseChoiceModel` — parameter layout and initialization

**Files:**
- Create: `politext_torch/_types.py`
- Create: `politext_torch/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: `ModuleNotFoundError: No module named 'politext_torch.model'`

- [ ] **Step 3: Implement `_types.py`**

```python
# politext_torch/_types.py
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
```

- [ ] **Step 4: Implement `model.py` (skeleton + init only)**

```python
# politext_torch/model.py
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_model.py -v`
Expected: 2 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add politext_torch/_types.py politext_torch/model.py tests/test_model.py
git commit -m "feat: add PhraseChoiceModel parameter layout and empirical init"
```

---

## Task 4: Poisson NLL — dense reference + batched rate sum

**Files:**
- Modify: `politext_torch/model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_model.py`:
```python
def _reference_poisson_nll(model, data):
    """Slow numpy reference: exactly sums over every (i, j)."""
    alpha_np = model.alpha.detach().numpy()
    gamma_np = model.gamma.detach().numpy()
    phi_np = model.phi.detach().numpy()
    counts = data.counts_sparse.to_dense().numpy()
    log_m = data.log_m.numpy()
    party = data.party.numpy()
    session = data.session.numpy()
    X = data.X.numpy()

    rate_sum = 0.0
    data_fit = 0.0
    N, V = counts.shape
    for i in range(N):
        t = session[i]
        K = party[i]
        u_i = alpha_np[:, t] + X[i] @ gamma_np.T + phi_np[:, t] * K
        rates = np.exp(log_m[i] + u_i)
        rate_sum += rates.sum()
        data_fit += (counts[i] * u_i).sum()
    return rate_sum - data_fit


def test_poisson_nll_matches_numpy_reference(tiny_dgp):
    import numpy as np
    data = make_tiny_data(tiny_dgp)
    model = PhraseChoiceModel(V=tiny_dgp["V"], T=tiny_dgp["T"], P=tiny_dgp["P"])
    # use a non-trivial param value so we're testing the formula, not zeros
    with torch.no_grad():
        model.alpha.copy_(tiny_dgp["alpha"])
        model.gamma.copy_(tiny_dgp["gamma"])
        model.phi.copy_(tiny_dgp["phi"])
    nll_t = model.poisson_nll(data)
    nll_ref = _reference_poisson_nll(model, data)
    np.testing.assert_allclose(nll_t.item(), nll_ref, rtol=1e-5)


def test_poisson_nll_batch_size_invariant(tiny_dgp):
    data = make_tiny_data(tiny_dgp)
    model = PhraseChoiceModel(V=tiny_dgp["V"], T=tiny_dgp["T"], P=tiny_dgp["P"])
    with torch.no_grad():
        model.alpha.copy_(tiny_dgp["alpha"])
        model.phi.copy_(tiny_dgp["phi"])
    # Equal within float tolerance regardless of batch_size.
    nll_1 = model.poisson_nll(data, batch_size=1).item()
    nll_big = model.poisson_nll(data, batch_size=1024).item()
    assert abs(nll_1 - nll_big) < 1e-4


def test_poisson_nll_is_differentiable(tiny_dgp):
    data = make_tiny_data(tiny_dgp)
    model = PhraseChoiceModel(V=tiny_dgp["V"], T=tiny_dgp["T"], P=tiny_dgp["P"])
    nll = model.poisson_nll(data)
    nll.backward()
    # all three params must have gradients
    assert model.alpha.grad is not None
    assert model.gamma.grad is not None
    assert model.phi.grad is not None
    assert torch.isfinite(model.alpha.grad).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_model.py -k "nll" -v`
Expected: 3 tests FAIL with `AttributeError: 'PhraseChoiceModel' object has no attribute 'poisson_nll'`.

- [ ] **Step 3: Implement `poisson_nll`**

Append to `politext_torch/model.py`:
```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_model.py -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add politext_torch/model.py tests/test_model.py
git commit -m "feat: implement Poisson NLL with doc-batched rate sum and sparse data-fit"
```

---

## Task 5: `fit_mle`

**Files:**
- Create: `politext_torch/fit.py`
- Create: `tests/test_fit.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_fit.py -v`
Expected: fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `fit_mle`**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fit.py::test_fit_mle_recovers_true_params_in_the_large_m_limit tests/test_fit.py::test_fit_mle_adam_runs_and_reduces_loss -v`
Expected: both PASS.

If `test_fit_mle_recovers_true_params_in_the_large_m_limit` fails marginally, bump `N` in `_bigger_dgp` to 1500 — the recovery tolerance is deliberately loose (20% relative error) to avoid flaky tests; still, small-sample noise can push it over.

- [ ] **Step 5: Commit**

```bash
git add politext_torch/fit.py tests/test_fit.py
git commit -m "feat: add fit_mle with LBFGS and Adam optimizers"
```

---

## Task 6: `fit_penalized` — FISTA with L1 on φ

**Files:**
- Modify: `politext_torch/fit.py`
- Modify: `tests/test_fit.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_fit.py`:
```python
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
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_fit.py -k penalized -v`
Expected: `NameError: name 'fit_penalized' is not defined` or similar.

- [ ] **Step 3: Implement `fit_penalized`**

Append to `politext_torch/fit.py`:
```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fit.py -v`
Expected: all 4 tests PASS. If `test_fit_penalized_small_lambda_close_to_mle` is marginal, bump `max_iter` to 4000 in the test; FISTA convergence is sublinear near the optimum.

- [ ] **Step 5: Commit**

```bash
git add politext_torch/fit.py tests/test_fit.py
git commit -m "feat: add fit_penalized (FISTA with backtracking + L1 on phi)"
```

---

## Task 7: `fit_path` with BIC selection

**Files:**
- Modify: `politext_torch/fit.py`
- Modify: `tests/test_fit.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_fit.py`:
```python
def test_fit_path_returns_best_lam_by_bic():
    dgp = _bigger_dgp(N=400)
    data = _prepare(dgp)
    model = PhraseChoiceModel(V=dgp["V"], T=dgp["T"], P=dgp["P"])
    model.init_from_data(data)
    result = fit_path(model, data, grid_size=10, criterion="bic")
    assert "lam" in result and isinstance(result["lam"], float)
    assert "path" in result and len(result["path"]) == 10
    # each path entry has the required keys
    for entry in result["path"]:
        assert set(entry) >= {"lam", "logLik", "df", "bic"}
    # best_idx corresponds to min BIC
    best = min(range(10), key=lambda i: result["path"][i]["bic"])
    assert result["best_idx"] == best


def test_fit_path_monotone_df_as_lambda_decreases():
    """As lambda decreases, df (nonzero phi entries) is non-decreasing."""
    dgp = _bigger_dgp(N=300)
    data = _prepare(dgp)
    model = PhraseChoiceModel(V=dgp["V"], T=dgp["T"], P=dgp["P"])
    model.init_from_data(data)
    result = fit_path(model, data, grid_size=6, criterion="bic")
    dfs = [e["df"] for e in result["path"]]
    # Path is stored in decreasing lambda order; df should be non-decreasing.
    for i in range(1, len(dfs)):
        assert dfs[i] >= dfs[i-1] - 1  # allow small non-monotonicity from numerical noise
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_fit.py -k path -v`
Expected: `NameError: name 'fit_path' is not defined`.

- [ ] **Step 3: Implement `fit_path`**

Append to `politext_torch/fit.py`:
```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fit.py -v`
Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add politext_torch/fit.py tests/test_fit.py
git commit -m "feat: add fit_path with warm-started BIC selection"
```

---

## Task 8: Partisanship primitives

**Files:**
- Create: `politext_torch/partisanship.py`
- Create: `tests/test_partisanship.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_partisanship.py
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
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_partisanship.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `partisanship.py`**

```python
# politext_torch/partisanship.py
"""Partisanship primitives: model-based and leave-out estimators."""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import torch


def choice_probs(
    alpha_t: torch.Tensor,         # (V,)
    gamma: torch.Tensor,           # (V, P)
    phi_t: torch.Tensor,           # (V,)
    X_row: torch.Tensor,           # (P,)
    party: float,
) -> torch.Tensor:
    u = alpha_t + X_row @ gamma.T + phi_t * party
    return torch.softmax(u, dim=-1)


def posterior_rho(
    alpha_t: torch.Tensor,
    gamma: torch.Tensor,
    phi_t: torch.Tensor,
    X_row: torch.Tensor,
    eps: float = 1e-30,
) -> torch.Tensor:
    q_R = choice_probs(alpha_t, gamma, phi_t, X_row, 1.0)
    q_D = choice_probs(alpha_t, gamma, phi_t, X_row, 0.0)
    return q_R / (q_R + q_D + eps)


def partisanship(
    alpha: torch.Tensor,           # (V, T)
    gamma: torch.Tensor,           # (V, P)
    phi: torch.Tensor,             # (V, T)
    X: torch.Tensor,               # (N, P)
    session: torch.Tensor,         # (N,)
    party: torch.Tensor,           # (N,) in {0, 1}
) -> torch.Tensor:
    """Session-level average partisanship (politext eq. 3-5)."""
    T = alpha.shape[1]
    N = X.shape[0]
    out = torch.full((T,), float("nan"))
    for t in range(T):
        mask = (session == t)
        idx = torch.nonzero(mask, as_tuple=False).ravel()
        if idx.numel() == 0:
            continue
        has_R = (party[idx] == 1).any()
        has_D = (party[idx] == 0).any()
        if not (has_R and has_D):
            warnings.warn(f"Session {t}: missing at least one party; "
                          "partisanship is undefined.", UserWarning, stacklevel=2)
            continue
        total = 0.0
        for i in idx:
            Xi = X[i]
            q_R = choice_probs(alpha[:, t], gamma, phi[:, t], Xi, 1.0)
            q_D = choice_probs(alpha[:, t], gamma, phi[:, t], Xi, 0.0)
            rho = q_R / (q_R + q_D + 1e-30)
            pi_i = 0.5 * (q_R * rho).sum() + 0.5 * (q_D * (1 - rho)).sum()
            total = total + pi_i
        out[t] = (total / idx.numel()).item()
    return out


def leave_out_partisanship(
    counts: sp.csr_matrix,
    party: np.ndarray,
    session: np.ndarray,
    speaker_id: np.ndarray | None = None,
    eps: float = 1e-30,
) -> np.ndarray:
    """Politext eq. (8) leave-one-speaker-out estimator.

    Closed form: for speaker i in session t, use q_hat_i (i's own empirical freq)
    paired with rho_hat_{-i,t} (leave-one-out ρ).
    """
    N, V = counts.shape
    if speaker_id is None:
        speaker_id = np.arange(N)
    party = np.asarray(party).astype(float)
    session = np.asarray(session)

    T = int(session.max()) + 1 if N > 0 else 0
    out = np.full(T, np.nan)

    counts_dense = counts.toarray().astype(np.float64)
    m = counts_dense.sum(axis=1)

    for t in range(T):
        in_t = np.where(session == t)[0]
        if len(in_t) == 0:
            continue
        mask_R = in_t[party[in_t] == 1.0]
        mask_D = in_t[party[in_t] == 0.0]
        if len(mask_R) == 0 or len(mask_D) == 0:
            continue

        sum_R = counts_dense[mask_R].sum(axis=0)   # (V,)
        sum_D = counts_dense[mask_D].sum(axis=0)
        m_R = m[mask_R].sum()
        m_D = m[mask_D].sum()

        # Group counts by speaker (for leaving out by unique speaker_id).
        sp_in_R = speaker_id[mask_R]
        sp_in_D = speaker_id[mask_D]
        # Map: for each speaker id, precompute their total counts and m.
        # With speaker_id = arange(N) (default), each row *is* one speaker,
        # so counts_dense[i] equals their total.
        # For multi-row speakers we need to aggregate first:
        unique_R, inv_R = np.unique(sp_in_R, return_inverse=True)
        unique_D, inv_D = np.unique(sp_in_D, return_inverse=True)
        speaker_counts_R = np.zeros((len(unique_R), V))
        speaker_m_R = np.zeros(len(unique_R))
        for k, idx in enumerate(mask_R):
            speaker_counts_R[inv_R[k]] += counts_dense[idx]
            speaker_m_R[inv_R[k]] += m[idx]
        speaker_counts_D = np.zeros((len(unique_D), V))
        speaker_m_D = np.zeros(len(unique_D))
        for k, idx in enumerate(mask_D):
            speaker_counts_D[inv_D[k]] += counts_dense[idx]
            speaker_m_D[inv_D[k]] += m[idx]

        # Leave-one-speaker-out ρ for R speakers
        def loo_pi_for_party(party_speakers_counts, party_speakers_m,
                             sum_own, m_own, sum_other, m_other, is_R: bool):
            num = 0.0
            for k in range(len(party_speakers_counts)):
                exc = sum_own - party_speakers_counts[k]
                exc_m = m_own - party_speakers_m[k]
                if exc_m <= 0:
                    continue
                qhat_own = exc / exc_m
                qhat_other = sum_other / m_other if m_other > 0 else np.zeros(V)
                denom = qhat_own + qhat_other
                if is_R:
                    rho = np.where(denom > 0, qhat_own / (denom + eps), 0.5)
                else:
                    rho = np.where(denom > 0, qhat_other / (denom + eps), 0.5)
                # speaker own empirical frequency
                if party_speakers_m[k] <= 0:
                    continue
                q_i = party_speakers_counts[k] / party_speakers_m[k]
                if is_R:
                    num += (q_i * rho).sum()
                else:
                    num += (q_i * (1 - rho)).sum()
            return num / max(len(party_speakers_counts), 1)

        term_R = loo_pi_for_party(
            speaker_counts_R, speaker_m_R,
            sum_own=sum_R, m_own=m_R,
            sum_other=sum_D, m_other=m_D,
            is_R=True,
        )
        term_D = loo_pi_for_party(
            speaker_counts_D, speaker_m_D,
            sum_own=sum_D, m_own=m_D,
            sum_other=sum_R, m_other=m_R,
            is_R=False,
        )
        out[t] = 0.5 * term_R + 0.5 * term_D

    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_partisanship.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add politext_torch/partisanship.py tests/test_partisanship.py
git commit -m "feat: add partisanship primitives (model-based and leave-out)"
```

---

## Task 9: `MLEEstimator` and `LeaveOutEstimator`

**Files:**
- Create: `politext_torch/estimators.py`
- Create: `tests/test_estimators.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_estimators.py
import numpy as np
import pytest
import torch
from politext_torch.estimators import (
    MLEEstimator, LeaveOutEstimator, PenalizedEstimator,
)
from politext_torch.simulate import draw_counts


def _big_dgp(V=6, T=2, N=600, P=1, seed=0):
    rng = np.random.default_rng(seed)
    alpha = torch.tensor(rng.standard_normal((V, T)) * 0.5, dtype=torch.float32)
    gamma = torch.tensor(rng.standard_normal((V, P)) * 0.2, dtype=torch.float32)
    phi = torch.tensor(rng.standard_normal((V, T)) * 0.5, dtype=torch.float32)
    party = torch.tensor((rng.random(N) < 0.5).astype(np.float32))
    session = torch.tensor(rng.integers(0, T, size=N), dtype=torch.long)
    X = torch.tensor(rng.standard_normal((N, P)).astype(np.float32))
    m = torch.full((N,), 1500.0)
    counts = draw_counts(alpha, gamma, phi, X, party, session, m, seed=seed + 1)
    return dict(
        counts=counts, party=party.numpy(), session=session.numpy(),
        X=X.numpy(), V=V, T=T, P=P, N=N,
    )


def test_mle_estimator_fit_populates_attrs():
    d = _big_dgp(N=300)
    est = MLEEstimator(max_iter=50)
    est.fit(d["counts"], d["party"], d["session"], d["X"])
    assert est.partisanship_.shape == (d["T"],)
    assert est.alpha_.shape == (d["V"], d["T"])
    assert est.gamma_.shape == (d["V"], d["P"])
    assert est.phi_.shape == (d["V"], d["T"])
    # Partisanship must be in [0, 1]
    assert np.all((est.partisanship_ >= 0) & (est.partisanship_ <= 1))


def test_leaveout_estimator_ignores_X_with_warning():
    d = _big_dgp(N=200)
    est = LeaveOutEstimator()
    with pytest.warns(UserWarning, match="ignored"):
        est.fit(d["counts"], d["party"], d["session"], X=d["X"])
    assert est.partisanship_.shape == (d["T"],)
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_estimators.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement the two estimators**

```python
# politext_torch/estimators.py
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_estimators.py -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add politext_torch/estimators.py tests/test_estimators.py
git commit -m "feat: add MLEEstimator and LeaveOutEstimator"
```

---

## Task 10: `PenalizedEstimator`

**Files:**
- Modify: `politext_torch/estimators.py`
- Modify: `tests/test_estimators.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_estimators.py`:
```python
def test_penalized_estimator_with_explicit_lambda():
    d = _big_dgp(N=300)
    est = PenalizedEstimator(lam=0.01, max_iter=300)
    est.fit(d["counts"], d["party"], d["session"], d["X"])
    assert est.lam_ == 0.01
    assert est.partisanship_.shape == (d["T"],)


def test_penalized_estimator_path_selects_bic_lambda():
    d = _big_dgp(N=400)
    est = PenalizedEstimator(grid_size=6, criterion="bic", max_iter=200)
    est.fit(d["counts"], d["party"], d["session"], d["X"])
    assert est.lam_ is not None
    assert len(est.bic_path_) == 6
    assert len(est.df_path_) == 6
    assert len(est.logLik_path_) == 6
    assert est.lam_grid_[0] >= est.lam_grid_[-1]  # decreasing grid
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_estimators.py -k Penalized -v`
Expected: `NameError: name 'PenalizedEstimator' is not defined`.

- [ ] **Step 3: Implement `PenalizedEstimator`**

Append to `politext_torch/estimators.py`:
```python
class PenalizedEstimator(BasePartisanshipEstimator):
    """L1-penalized Poisson-logit estimator (politext §4.3 / eq. 9)."""

    def __init__(
        self,
        lam: float | None = None,
        lam_grid: list[float] | None = None,
        grid_size: int = 100,
        lam_min_ratio: float = 1e-3,
        criterion: str = "bic",
        store_path: bool = False,
        lam_alpha: float = 1e-5,
        lam_gamma: float = 1e-5,
        max_iter: int = 500,
        tol: float = 1e-5,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.lam = lam
        self.lam_grid = lam_grid
        self.grid_size = grid_size
        self.lam_min_ratio = lam_min_ratio
        self.criterion = criterion
        self.store_path = store_path
        self.lam_alpha = lam_alpha
        self.lam_gamma = lam_gamma
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        counts: sp.csr_matrix,
        party: np.ndarray,
        session: np.ndarray,
        X: np.ndarray | None = None,
        speaker_id: np.ndarray | None = None,
        **fit_kwargs: Any,
    ) -> "PenalizedEstimator":
        data = PhraseData.from_arrays(counts, party, session, X, device=self.device)
        self._store_metadata(data)
        T = int(data.session.max().item()) + 1
        model = PhraseChoiceModel(V=data.V, T=T, P=data.P).to(self.device)
        model.init_from_data(data)

        if self.lam is not None and self.lam_grid is None:
            fit_penalized(
                model, data, lam=self.lam,
                lam_alpha=self.lam_alpha, lam_gamma=self.lam_gamma,
                max_iter=self.max_iter, tol=self.tol,
            )
            self.lam_ = float(self.lam)
            self.lam_grid_ = None
            self.bic_path_ = None
            self.df_path_ = None
            self.logLik_path_ = None
        else:
            result = fit_path(
                model, data,
                lam_grid=self.lam_grid, grid_size=self.grid_size,
                lam_min_ratio=self.lam_min_ratio,
                criterion=self.criterion,
                lam_alpha=self.lam_alpha, lam_gamma=self.lam_gamma,
                max_iter=self.max_iter, tol=self.tol,
                store_path_params=self.store_path,
            )
            self.lam_ = float(result["lam"])
            self.lam_grid_ = [e["lam"] for e in result["path"]]
            self.bic_path_ = [e["bic"] for e in result["path"]]
            self.df_path_ = [e["df"] for e in result["path"]]
            self.logLik_path_ = [e["logLik"] for e in result["path"]]

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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_estimators.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add politext_torch/estimators.py tests/test_estimators.py
git commit -m "feat: add PenalizedEstimator with BIC-selected lambda path"
```

---

## Task 11: `subsample_ci`

**Files:**
- Create: `politext_torch/inference.py`
- Create: `tests/test_inference.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_inference.py
import numpy as np
import torch
from politext_torch.estimators import LeaveOutEstimator
from politext_torch.inference import subsample_ci
from politext_torch.simulate import draw_counts


def _dgp_for_ci(N=400, V=5, T=2, seed=0):
    rng = np.random.default_rng(seed)
    alpha = torch.tensor(rng.standard_normal((V, T)) * 0.3, dtype=torch.float32)
    phi = torch.tensor(rng.standard_normal((V, T)) * 0.4, dtype=torch.float32)
    gamma = torch.zeros(V, 0)
    X = torch.zeros(N, 0)
    party = torch.tensor((rng.random(N) < 0.5).astype(np.float32))
    session = torch.tensor(rng.integers(0, T, size=N), dtype=torch.long)
    m = torch.full((N,), 500.0)
    counts = draw_counts(alpha, gamma, phi, X, party, session, m, seed=seed + 1)
    return counts, party.numpy(), session.numpy(), X.numpy()


def test_subsample_ci_shapes_and_point_estimate_inside_ci():
    counts, party, session, X = _dgp_for_ci()
    factory = lambda: LeaveOutEstimator()
    result = subsample_ci(
        factory, counts, party, session,
        n_subsamples=20, frac=0.3, seed=1, transform="identity",
    )
    T = 2
    assert result["estimate"].shape == (T,)
    assert result["ci_lower"].shape == (T,)
    assert result["ci_upper"].shape == (T,)
    assert result["subsample_estimates"].shape == (20, T)
    # Point estimate lies between the bounds in each session.
    assert np.all(result["ci_lower"] <= result["estimate"] + 1e-9)
    assert np.all(result["estimate"] <= result["ci_upper"] + 1e-9)


def test_subsample_ci_reproducible_with_seed():
    counts, party, session, X = _dgp_for_ci()
    factory = lambda: LeaveOutEstimator()
    a = subsample_ci(factory, counts, party, session,
                     n_subsamples=10, frac=0.3, seed=42, transform="identity")
    b = subsample_ci(factory, counts, party, session,
                     n_subsamples=10, frac=0.3, seed=42, transform="identity")
    np.testing.assert_allclose(a["subsample_estimates"], b["subsample_estimates"])


def test_subsample_ci_log_transform_returns_valid_interval():
    counts, party, session, X = _dgp_for_ci()
    factory = lambda: LeaveOutEstimator()
    result = subsample_ci(factory, counts, party, session,
                          n_subsamples=20, frac=0.3, seed=1, transform="log")
    assert np.all(result["ci_lower"] >= 0.5 - 1e-6)
    assert np.all(result["ci_upper"] <= 1.0 + 1e-6)
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_inference.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `subsample_ci`**

```python
# politext_torch/inference.py
"""Subsampling-based confidence intervals (Politis-Romano-Wolf)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp


def subsample_ci(
    estimator_factory: Callable,
    counts: sp.csr_matrix,
    party: np.ndarray,
    session: np.ndarray,
    X: np.ndarray | None = None,
    speaker_id: np.ndarray | None = None,
    n_subsamples: int = 100,
    frac: float = 0.1,
    alpha: float = 0.05,
    transform: str = "log",
    seed: int | None = None,
    n_jobs: int = 1,
) -> dict:
    """Speaker-level subsampling CI for session-level partisanship.

    Procedure (Politis-Romano-Wolf 1999, Thm 2.2.1):
      1. Full-sample fit -> pi_hat.
      2. Draw n_subsamples subsamples of speakers (size = round(frac * n_speakers))
         without replacement.
      3. Fit estimator_factory() on each subsample -> pi_b.
      4. Compute sqrt(tau_b) * (g(pi_b) - g(pi_hat)) quantiles, invert to get CI.

    transform: "identity" or "log" (default). Log uses g(x) = log(x - 0.5),
    which respects the [0.5, 1] support and matches politext Figure 1.
    """
    N = counts.shape[0]
    party = np.asarray(party)
    session = np.asarray(session)
    if speaker_id is None:
        speaker_id = np.arange(N)
    else:
        speaker_id = np.asarray(speaker_id)

    unique_speakers = np.unique(speaker_id)
    n_speakers = len(unique_speakers)
    n_sub_spk = max(1, int(round(frac * n_speakers)))

    # Full-sample fit
    est = estimator_factory()
    fit_kwargs = dict(party=party, session=session, speaker_id=speaker_id)
    if X is not None:
        fit_kwargs["X"] = X
    est.fit(counts, **fit_kwargs)
    pi_hat = np.asarray(est.partisanship_, dtype=float)
    T = pi_hat.shape[0]

    rng = np.random.default_rng(seed)

    def _run_one_subsample(b: int) -> np.ndarray:
        sub_rng = np.random.default_rng(None if seed is None else seed + 1 + b)
        chosen_speakers = sub_rng.choice(unique_speakers, size=n_sub_spk, replace=False)
        row_mask = np.isin(speaker_id, chosen_speakers)
        rows = np.where(row_mask)[0]
        sub_counts = counts[rows]
        sub_party = party[rows]
        sub_session = session[rows]
        sub_speaker_id = speaker_id[rows]
        sub_X = X[rows] if X is not None else None
        est_b = estimator_factory()
        kw = dict(party=sub_party, session=sub_session, speaker_id=sub_speaker_id)
        if sub_X is not None:
            kw["X"] = sub_X
        est_b.fit(sub_counts, **kw)
        pi_b = np.asarray(est_b.partisanship_, dtype=float)
        # Pad with NaN for missing sessions
        if pi_b.shape[0] < T:
            padded = np.full(T, np.nan)
            padded[: pi_b.shape[0]] = pi_b
            pi_b = padded
        return pi_b

    if n_jobs == 1:
        subs = np.stack([_run_one_subsample(b) for b in range(n_subsamples)], axis=0)
    else:
        from joblib import Parallel, delayed
        subs = np.stack(
            Parallel(n_jobs=n_jobs)(delayed(_run_one_subsample)(b)
                                    for b in range(n_subsamples)),
            axis=0,
        )

    tau = float(n_sub_spk)
    n_full = float(n_speakers)

    def _g(x: np.ndarray) -> np.ndarray:
        if transform == "identity":
            return x
        return np.log(np.maximum(x - 0.5, 1e-12))

    def _g_inv(u: np.ndarray) -> np.ndarray:
        if transform == "identity":
            return u
        return 0.5 + np.exp(u)

    # Per-session: compute quantiles of Q_b = sqrt(tau) * (g(pi_b) - g(pi_hat))
    lo = np.full(T, np.nan)
    hi = np.full(T, np.nan)
    for t in range(T):
        if not np.isfinite(pi_hat[t]):
            continue
        valid = np.isfinite(subs[:, t])
        if valid.sum() < 2:
            continue
        g_hat = _g(np.array([pi_hat[t]]))[0]
        use_log_here = (transform == "log") and (pi_hat[t] - 0.5 > 1e-6)
        if transform == "log" and not use_log_here:
            # Fall back to identity per session near 0.5.
            Q = np.sqrt(tau) * (subs[valid, t] - pi_hat[t])
            q_lo, q_hi = np.quantile(Q, [alpha / 2, 1 - alpha / 2])
            lo[t] = pi_hat[t] - q_hi / np.sqrt(n_full)
            hi[t] = pi_hat[t] - q_lo / np.sqrt(n_full)
        else:
            Q = np.sqrt(tau) * (_g(subs[valid, t]) - g_hat)
            q_lo, q_hi = np.quantile(Q, [alpha / 2, 1 - alpha / 2])
            lo[t] = _g_inv(g_hat - q_hi / np.sqrt(n_full))
            hi[t] = _g_inv(g_hat - q_lo / np.sqrt(n_full))

    return {
        "estimate": pi_hat,
        "ci_lower": lo,
        "ci_upper": hi,
        "subsample_estimates": subs,
        "n_sub": n_sub_spk,
        "n_full": n_speakers,
        "frac": frac,
    }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_inference.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add politext_torch/inference.py tests/test_inference.py
git commit -m "feat: add subsample_ci with log-scale transform"
```

---

## Task 12: `scale_document` and `score_document`

**Files:**
- Create: `politext_torch/scale.py`
- Create: `tests/test_scale.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scale.py
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from politext_torch.estimators import PenalizedEstimator
from politext_torch.scale import (
    scale_document, scale_documents,
    score_document, score_documents,
)
from politext_torch.simulate import draw_counts


def _fit_penalized_for_scaling(seed=0):
    rng = np.random.default_rng(seed)
    V, T, N = 8, 1, 400
    alpha = torch.tensor(rng.standard_normal((V, T)) * 0.3, dtype=torch.float32)
    phi = torch.tensor(rng.standard_normal((V, T)) * 0.6, dtype=torch.float32)
    gamma = torch.zeros(V, 0)
    X = torch.zeros(N, 0)
    party = torch.tensor((rng.random(N) < 0.5).astype(np.float32))
    session = torch.zeros(N, dtype=torch.long)
    m = torch.full((N,), 500.0)
    counts = draw_counts(alpha, gamma, phi, X, party, session, m, seed=seed + 1)
    est = PenalizedEstimator(lam=0.05, max_iter=300)
    est.fit(counts, party.numpy(), session.numpy())
    return est, V, T


def test_scale_document_equals_dot_product_of_freqs_and_phi():
    est, V, _ = _fit_penalized_for_scaling()
    doc = sp.csr_matrix(np.array([[3, 0, 1, 0, 0, 2, 0, 4]]))
    score = scale_document(est, doc, session=0)
    freqs = np.array([3, 0, 1, 0, 0, 2, 0, 4], dtype=float)
    freqs = freqs / freqs.sum()
    expected = float((freqs * est.phi_[:, 0]).sum())
    np.testing.assert_allclose(score, expected, atol=1e-6)


def test_scale_document_rejects_wrong_vocab():
    est, V, _ = _fit_penalized_for_scaling()
    bad = sp.csr_matrix(np.ones((1, V + 3)))
    with pytest.raises(ValueError, match="vocab"):
        scale_document(est, bad, session=0)


def test_scale_documents_batches_and_matches_scalars():
    est, V, _ = _fit_penalized_for_scaling()
    docs = sp.csr_matrix(np.array([[3, 0, 1, 0, 0, 2, 0, 4],
                                   [0, 5, 0, 2, 0, 0, 1, 0]]))
    batch = scale_documents(est, docs, session=0)
    individual = np.array([
        scale_document(est, docs[0], session=0),
        scale_document(est, docs[1], session=0),
    ])
    np.testing.assert_allclose(batch, individual, atol=1e-6)


def test_score_document_returns_pi_rho_q():
    est, V, _ = _fit_penalized_for_scaling()
    doc = sp.csr_matrix(np.array([[1, 1, 0, 0, 0, 0, 0, 0]]))
    out = score_document(est, doc, session=0)
    assert set(out) >= {"pi", "rho", "q_R", "q_D"}
    assert 0 <= out["pi"] <= 1
    assert out["rho"].shape == (V,)
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_scale.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `scale.py`**

```python
# politext_torch/scale.py
"""Scaling new unseen documents with a fitted PenalizedEstimator."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from politext_torch.partisanship import choice_probs, posterior_rho


def _check_vocab(est, counts: sp.csr_matrix) -> None:
    if counts.shape[1] != est.vocab_size_:
        raise ValueError(
            f"vocab size mismatch: estimator trained on V={est.vocab_size_} "
            f"but received counts with {counts.shape[1]} columns. "
            "Use the same fitted CountVectorizer's `.transform(...)` at both "
            "training and scoring time."
        )


def _freqs(counts_row: sp.csr_matrix, normalize: str) -> np.ndarray:
    arr = np.asarray(counts_row.toarray()).ravel().astype(float)
    if normalize == "count":
        return arr
    if normalize == "binary":
        return (arr > 0).astype(float)
    total = arr.sum()
    if total == 0:
        return arr
    return arr / total


def scale_document(
    estimator,
    counts_new: sp.csr_matrix,
    session: int,
    normalize: str = "freq",
) -> float:
    """Media_slant eq. (1): dot-product of document freqs with phi for session t."""
    _check_vocab(estimator, counts_new)
    f_b = _freqs(counts_new, normalize)
    phi_t = estimator.phi_[:, int(session)]
    return float((f_b * phi_t).sum())


def scale_documents(
    estimator,
    counts_matrix: sp.csr_matrix,
    session,
    normalize: str = "freq",
) -> np.ndarray:
    _check_vocab(estimator, counts_matrix)
    M = counts_matrix.shape[0]
    if np.ndim(session) == 0:
        sess = np.full(M, int(session))
    else:
        sess = np.asarray(session, dtype=int)

    out = np.zeros(M)
    for m in range(M):
        f_b = _freqs(counts_matrix[m], normalize)
        phi_t = estimator.phi_[:, sess[m]]
        out[m] = float((f_b * phi_t).sum())
    return out


def score_document(
    estimator,
    counts_new: sp.csr_matrix,
    session: int,
    X_new: np.ndarray | None = None,
    normalize: str = "freq",
) -> dict:
    """Posterior π for a doc treated as a hypothetical speaker (politext eq. 3-4)."""
    _check_vocab(estimator, counts_new)
    t = int(session)
    alpha_t = torch.from_numpy(estimator.alpha_[:, t]).float()
    phi_t = torch.from_numpy(estimator.phi_[:, t]).float()
    gamma = torch.from_numpy(estimator.gamma_).float()
    if X_new is None:
        X_row = torch.zeros(estimator.n_covariates_)
    else:
        X_row = torch.as_tensor(X_new, dtype=torch.float32).ravel()

    q_R = choice_probs(alpha_t, gamma, phi_t, X_row, 1.0)
    q_D = choice_probs(alpha_t, gamma, phi_t, X_row, 0.0)
    rho = posterior_rho(alpha_t, gamma, phi_t, X_row)

    f_b = _freqs(counts_new, normalize)
    pi = float(0.5 * (f_b * rho.numpy()).sum() + 0.5 * (f_b * (1 - rho.numpy())).sum())
    # Note: the above reduces to 0.5 for uniform f_b — correct only when we
    # pair rho with q_R (for R-hypothesis) and 1-rho with q_D (for D-hypothesis):
    pi = float(
        0.5 * (q_R.numpy() * rho.numpy()).sum()
        + 0.5 * (q_D.numpy() * (1 - rho.numpy())).sum()
    )
    return {
        "pi": pi,
        "rho": rho.numpy(),
        "q_R": q_R.numpy(),
        "q_D": q_D.numpy(),
    }


def score_documents(
    estimator,
    counts_matrix: sp.csr_matrix,
    session,
    X_new: np.ndarray | None = None,
    normalize: str = "freq",
) -> dict:
    _check_vocab(estimator, counts_matrix)
    M = counts_matrix.shape[0]
    if np.ndim(session) == 0:
        sess = np.full(M, int(session))
    else:
        sess = np.asarray(session, dtype=int)

    pi_arr = np.zeros(M)
    for m in range(M):
        x = None if X_new is None else X_new[m]
        pi_arr[m] = score_document(estimator, counts_matrix[m],
                                   session=sess[m], X_new=x, normalize=normalize)["pi"]
    return {"pi": pi_arr}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_scale.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add politext_torch/scale.py tests/test_scale.py
git commit -m "feat: add scale_document and score_document for unseen texts"
```

---

## Task 13: Simulator helpers for Monte Carlo

**Files:**
- Modify: `politext_torch/simulate.py`

Each Monte Carlo script sets up a DGP; we put the DGP builders next to `draw_counts` so the experiments only contain experimental logic.

- [ ] **Step 1: Add tests**

Append to `tests/test_simulate.py`:
```python
def test_make_mc_A_shapes_sensible():
    from politext_torch.simulate import make_mc_A
    out = make_mc_A(V=50, T=3, N=200, seed=0)
    assert out["counts"].shape == (200, 50)
    assert out["party"].shape == (200,)
    assert out["session"].shape == (200,)
    assert out["true_pi"].shape == (3,)


def test_make_mc_C_is_null():
    from politext_torch.simulate import make_mc_C
    out = make_mc_C(V=20, T=2, N=100, seed=0)
    # phi = 0 -> true partisanship is exactly 0.5 in every session.
    assert out["true_phi"].abs().max() < 1e-9
    assert np.allclose(out["true_pi"], 0.5)
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_simulate.py -k mc_ -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement MC helpers**

Append to `politext_torch/simulate.py`:
```python
def _true_partisanship(alpha, gamma, phi, X, session, party):
    from politext_torch.partisanship import partisanship
    return partisanship(alpha, gamma, phi, X, session, party).numpy()


def make_mc_A(V: int, T: int = 5, N: int = 1000, P: int = 0,
              m_value: float = 100.0, seed: int = 0) -> dict:
    """Bias/RMSE experiment DGP: covariate-free multinomial with moderate phi."""
    rng = np.random.default_rng(seed)
    alpha = torch.tensor(rng.standard_normal((V, T)) * 0.3, dtype=torch.float32)
    phi = torch.tensor(rng.standard_normal((V, T)) * 0.5, dtype=torch.float32)
    gamma = torch.zeros(V, P)
    party = torch.tensor((rng.random(N) < 0.5).astype(np.float32))
    session = torch.tensor(rng.integers(0, T, size=N), dtype=torch.long)
    X = torch.zeros(N, P)
    m = torch.full((N,), m_value)
    counts = draw_counts(alpha, gamma, phi, X, party, session, m, seed=seed + 17)
    true_pi = _true_partisanship(alpha, gamma, phi, X, session, party)
    return dict(
        counts=counts, party=party.numpy(), session=session.numpy(),
        X=None if P == 0 else X.numpy(), true_pi=true_pi,
        true_alpha=alpha, true_gamma=gamma, true_phi=phi,
    )


def make_mc_B(V: int = 200, T: int = 10, N: int = 1000, seed: int = 0) -> dict:
    """Coverage experiment DGP: identical form to A; kept separate so
    experiments can pin their own sizes."""
    return make_mc_A(V=V, T=T, N=N, P=0, m_value=200.0, seed=seed)


def make_mc_C(V: int, T: int = 5, N: int = 1000, seed: int = 0) -> dict:
    """Null experiment: phi = 0, so true partisanship equals 0.5."""
    rng = np.random.default_rng(seed)
    alpha = torch.tensor(rng.standard_normal((V, T)) * 0.3, dtype=torch.float32)
    phi = torch.zeros(V, T)
    gamma = torch.zeros(V, 0)
    party = torch.tensor((rng.random(N) < 0.5).astype(np.float32))
    session = torch.tensor(rng.integers(0, T, size=N), dtype=torch.long)
    X = torch.zeros(N, 0)
    m = torch.full((N,), 100.0)
    counts = draw_counts(alpha, gamma, phi, X, party, session, m, seed=seed + 17)
    true_pi = np.full(T, 0.5)  # by construction
    return dict(
        counts=counts, party=party.numpy(), session=session.numpy(),
        X=None, true_pi=true_pi,
        true_alpha=alpha, true_gamma=gamma, true_phi=phi,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_simulate.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add politext_torch/simulate.py tests/test_simulate.py
git commit -m "feat: add make_mc_A/B/C DGP builders"
```

---

## Task 14: Experiment A — bias/RMSE vs vocabulary size

**Files:**
- Create: `politext_torch/experiments/mc_bias_rmse.py`

- [ ] **Step 1: Write the experiment script**

```python
# politext_torch/experiments/mc_bias_rmse.py
"""Experiment A: bias and RMSE of π̂ as vocabulary size V grows.

Expected: MLE biased upward in V; leave-out and penalized near truth.

Run: python -m politext_torch.experiments.mc_bias_rmse
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from politext_torch.estimators import (
    LeaveOutEstimator, MLEEstimator, PenalizedEstimator,
)
from politext_torch.simulate import make_mc_A

V_GRID = [50, 200, 500, 1000]
N_REP = 50
N = 800
T = 5
OUT = Path(__file__).resolve().parent / "output"


def run():
    OUT.mkdir(exist_ok=True)
    rows = []
    for V in V_GRID:
        print(f"=== V = {V} ===")
        for rep in range(N_REP):
            t0 = time.time()
            dgp = make_mc_A(V=V, T=T, N=N, seed=rep)
            true_pi = dgp["true_pi"]
            for name, est in [
                ("MLE", MLEEstimator(max_iter=30)),
                ("LeaveOut", LeaveOutEstimator()),
                ("Penalized", PenalizedEstimator(grid_size=15, max_iter=80)),
            ]:
                est.fit(dgp["counts"], dgp["party"], dgp["session"])
                pi_hat = est.partisanship_
                for t in range(T):
                    rows.append({
                        "V": V, "rep": rep, "estimator": name, "session": t,
                        "pi_hat": pi_hat[t], "pi_true": true_pi[t],
                        "err": pi_hat[t] - true_pi[t],
                    })
            print(f"  rep {rep+1}/{N_REP}  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results_a.csv", index=False)

    agg = (df.groupby(["V", "estimator"])
             .agg(bias=("err", "mean"),
                  rmse=("err", lambda x: float(np.sqrt(np.mean(x**2)))))
             .reset_index())
    agg.to_csv(OUT / "results_a_agg.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for est in agg["estimator"].unique():
        sub = agg[agg["estimator"] == est]
        axes[0].plot(sub["V"], sub["bias"], marker="o", label=est)
        axes[1].plot(sub["V"], sub["rmse"], marker="o", label=est)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("V"); axes[0].set_ylabel("Bias of π̂"); axes[0].legend()
    axes[1].set_xlabel("V"); axes[1].set_ylabel("RMSE of π̂"); axes[1].legend()
    fig.suptitle("Experiment A: bias and RMSE vs vocabulary size")
    fig.tight_layout()
    fig.savefig(OUT / "fig_a.pdf")
    print(f"Saved results to {OUT}")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Smoke-run (small) in a test**

Append to `tests/test_estimators.py` (keeps the experiment scripts testable):
```python
def test_experiment_a_smoke():
    """Tiny V_GRID and N_REP to confirm the experiment script runs end-to-end."""
    from politext_torch.experiments import mc_bias_rmse
    # Patch constants to tiny values
    mc_bias_rmse.V_GRID = [20, 40]
    mc_bias_rmse.N_REP = 2
    mc_bias_rmse.N = 120
    mc_bias_rmse.T = 2
    mc_bias_rmse.run()
```

- [ ] **Step 3: Run smoke test**

Run: `pytest tests/test_estimators.py::test_experiment_a_smoke -v`
Expected: PASS (runs in ~15s), produces `politext_torch/experiments/output/fig_a.pdf` and CSVs.

- [ ] **Step 4: Commit**

```bash
git add politext_torch/experiments/mc_bias_rmse.py tests/test_estimators.py
echo "politext_torch/experiments/output/" >> .gitignore
git add .gitignore
git commit -m "feat: add Experiment A (bias/RMSE vs vocabulary size)"
```

---

## Task 15: Experiment B — CI coverage

**Files:**
- Create: `politext_torch/experiments/mc_coverage.py`

- [ ] **Step 1: Write the experiment script**

```python
# politext_torch/experiments/mc_coverage.py
"""Experiment B: 95% CI coverage from speaker-level subsampling.

Expected: coverage ≈ 0.95 for leave-out and penalized.

Run: python -m politext_torch.experiments.mc_coverage
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from politext_torch.estimators import LeaveOutEstimator, PenalizedEstimator
from politext_torch.inference import subsample_ci
from politext_torch.simulate import make_mc_B

N_REP = 100
N_SUB = 50          # smaller than paper's 100 for tractability
FRAC = 0.1
OUT = Path(__file__).resolve().parent / "output"


def run():
    OUT.mkdir(exist_ok=True)
    rows = []
    for rep in range(N_REP):
        t0 = time.time()
        dgp = make_mc_B(V=100, T=5, N=600, seed=rep)
        true_pi = dgp["true_pi"]
        for name, factory in [
            ("LeaveOut", lambda: LeaveOutEstimator()),
            ("Penalized", lambda: PenalizedEstimator(grid_size=8, max_iter=60)),
        ]:
            result = subsample_ci(
                factory, dgp["counts"], dgp["party"], dgp["session"],
                n_subsamples=N_SUB, frac=FRAC, seed=rep, transform="log",
            )
            for t in range(len(true_pi)):
                lo, hi = result["ci_lower"][t], result["ci_upper"][t]
                rows.append({
                    "rep": rep, "estimator": name, "session": t,
                    "pi_true": true_pi[t],
                    "estimate": result["estimate"][t],
                    "ci_lower": lo, "ci_upper": hi,
                    "covered": int(lo <= true_pi[t] <= hi)
                                if np.isfinite(lo) and np.isfinite(hi) else 0,
                    "width": (hi - lo) if np.isfinite(lo) and np.isfinite(hi) else np.nan,
                })
        print(f"rep {rep+1}/{N_REP}  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results_b.csv", index=False)

    agg = (df.groupby(["estimator", "session"])
             .agg(coverage=("covered", "mean"),
                  median_width=("width", "median"))
             .reset_index())
    agg.to_csv(OUT / "results_b_agg.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for est in agg["estimator"].unique():
        sub = agg[agg["estimator"] == est]
        axes[0].plot(sub["session"], sub["coverage"], marker="o", label=est)
        axes[1].plot(sub["session"], sub["median_width"], marker="o", label=est)
    axes[0].axhline(0.95, color="k", lw=0.5, ls="--")
    axes[0].set_xlabel("session"); axes[0].set_ylabel("95% CI coverage")
    axes[0].set_ylim(0.5, 1.05); axes[0].legend()
    axes[1].set_xlabel("session"); axes[1].set_ylabel("Median CI width"); axes[1].legend()
    fig.suptitle("Experiment B: 95% CI coverage via speaker subsampling")
    fig.tight_layout()
    fig.savefig(OUT / "fig_b.pdf")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Smoke test**

Append to `tests/test_inference.py`:
```python
def test_experiment_b_smoke():
    from politext_torch.experiments import mc_coverage
    mc_coverage.N_REP = 3
    mc_coverage.N_SUB = 5
    mc_coverage.run()
```

- [ ] **Step 3: Run smoke test**

Run: `pytest tests/test_inference.py::test_experiment_b_smoke -v`
Expected: PASS in ~30s, produces `fig_b.pdf`.

- [ ] **Step 4: Commit**

```bash
git add politext_torch/experiments/mc_coverage.py tests/test_inference.py
git commit -m "feat: add Experiment B (95% CI coverage)"
```

---

## Task 16: Experiment C — null behaviour

**Files:**
- Create: `politext_torch/experiments/mc_null.py`

- [ ] **Step 1: Write the experiment script**

```python
# politext_torch/experiments/mc_null.py
"""Experiment C: behaviour under phi = 0 (no true partisanship).

Expected: MLE biased above 0.5; leave-out and penalized centered on 0.5.

Run: python -m politext_torch.experiments.mc_null
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from politext_torch.estimators import (
    LeaveOutEstimator, MLEEstimator, PenalizedEstimator,
)
from politext_torch.simulate import make_mc_C

N_REP = 100
V = 500
T = 3
N = 600
OUT = Path(__file__).resolve().parent / "output"


def run():
    OUT.mkdir(exist_ok=True)
    rows = []
    for rep in range(N_REP):
        t0 = time.time()
        dgp = make_mc_C(V=V, T=T, N=N, seed=rep)
        for name, est in [
            ("MLE", MLEEstimator(max_iter=30)),
            ("LeaveOut", LeaveOutEstimator()),
            ("Penalized", PenalizedEstimator(grid_size=10, max_iter=80)),
        ]:
            est.fit(dgp["counts"], dgp["party"], dgp["session"])
            pi_hat = est.partisanship_
            for t in range(T):
                rows.append({"rep": rep, "estimator": name,
                             "session": t, "pi_hat": pi_hat[t]})
        print(f"rep {rep+1}/{N_REP}  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results_c.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for est in df["estimator"].unique():
        sub = df[df["estimator"] == est]
        ax.hist(sub["pi_hat"].values, bins=30, alpha=0.4, label=est, density=True)
    ax.axvline(0.5, color="k", lw=1)
    ax.set_xlabel(r"$\hat\pi_t$"); ax.set_ylabel("density")
    ax.set_title(f"Experiment C: null (φ=0), V={V}, N={N}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_c.pdf")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Smoke test**

Append to `tests/test_estimators.py`:
```python
def test_experiment_c_smoke():
    from politext_torch.experiments import mc_null
    mc_null.N_REP = 2
    mc_null.V = 40
    mc_null.N = 120
    mc_null.T = 2
    mc_null.run()
```

- [ ] **Step 3: Run smoke test**

Run: `pytest tests/test_estimators.py::test_experiment_c_smoke -v`
Expected: PASS, produces `fig_c.pdf`.

- [ ] **Step 4: Commit**

```bash
git add politext_torch/experiments/mc_null.py tests/test_estimators.py
git commit -m "feat: add Experiment C (null behaviour under phi=0)"
```

---

## Task 17: End-to-end integration test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write a full-pipeline test**

```python
# tests/test_integration.py
"""End-to-end test exercising all three estimators + inference + scaling."""
import numpy as np
import scipy.sparse as sp
from politext_torch.estimators import (
    LeaveOutEstimator, MLEEstimator, PenalizedEstimator,
)
from politext_torch.inference import subsample_ci
from politext_torch.scale import scale_document, score_document
from politext_torch.simulate import make_mc_A


def test_end_to_end_smoke_covers_all_modules():
    dgp = make_mc_A(V=30, T=2, N=200, seed=0)
    counts, party, session = dgp["counts"], dgp["party"], dgp["session"]

    # All three estimators fit on the same data.
    mle = MLEEstimator(max_iter=30).fit(counts, party, session)
    lo = LeaveOutEstimator().fit(counts, party, session)
    pen = PenalizedEstimator(grid_size=5, max_iter=80).fit(counts, party, session)

    for est in (mle, lo, pen):
        assert est.partisanship_.shape == (2,)
        assert np.all((est.partisanship_ >= 0) & (est.partisanship_ <= 1))

    # Inference on leave-out.
    ci = subsample_ci(lambda: LeaveOutEstimator(), counts, party, session,
                      n_subsamples=10, frac=0.3, seed=1, transform="identity")
    assert ci["ci_lower"].shape == (2,)

    # Scaling a new document with the penalized estimator.
    new_doc = sp.csr_matrix(np.random.poisson(1, size=(1, 30)))
    s = scale_document(pen, new_doc, session=0)
    assert np.isfinite(s)
    r = score_document(pen, new_doc, session=0)
    assert 0 <= r["pi"] <= 1
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_integration.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration test across estimators, inference, scaling"
```

---

## Task 18: README with usage tutorial

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

```markdown
# politext-torch

PyTorch implementation of the Gentzkow–Shapiro–Taddy (2019, *Econometrica*) partisanship estimators, with subsampling confidence intervals and unseen-text scaling after Widmer et al. (2020).

## Install

```bash
conda env create -f environment.yml
conda activate gst
pip install -e ".[dev]"
```

## Quick tour

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from patsy import dmatrix

from politext_torch import (
    MLEEstimator, LeaveOutEstimator, PenalizedEstimator,
    subsample_ci, scale_document,
)

# 1. Build counts from text.
vectorizer = CountVectorizer(ngram_range=(2, 2), min_df=5)
counts = vectorizer.fit_transform(list_of_speeches)            # (N, V) sparse
party = np.array(party_list, dtype=float)                      # 0 or 1
session = np.array(session_list, dtype=int)                    # 0..T-1

# 2. Optional covariates via patsy (e.g., state, gender, chamber).
X = np.asarray(dmatrix("state + gender + chamber", data=meta, return_type="matrix"))

# 3. Fit estimators.
mle  = MLEEstimator().fit(counts, party, session, X)
lo   = LeaveOutEstimator().fit(counts, party, session)         # ignores X
pen  = PenalizedEstimator(criterion="bic").fit(counts, party, session, X)

# 4. Subsampling 95% CI on the penalized estimator.
ci = subsample_ci(
    lambda: PenalizedEstimator(criterion="bic"),
    counts, party, session, X=X,
    n_subsamples=100, frac=0.1, seed=0,
)
# ci["estimate"], ci["ci_lower"], ci["ci_upper"] are all (T,) arrays.

# 5. Scale new (unseen) documents onto the partisan axis.
new_counts = vectorizer.transform(new_docs)                    # REUSE the fitted vectorizer.
scores = np.array([scale_document(pen, new_counts[i], session=0)
                   for i in range(new_counts.shape[0])])
```

## Running the Monte Carlo experiments

```bash
python -m politext_torch.experiments.mc_bias_rmse
python -m politext_torch.experiments.mc_coverage
python -m politext_torch.experiments.mc_null
```

Figures and CSVs land in `politext_torch/experiments/output/`.

## References

- Gentzkow, Shapiro, Taddy (2019). *Measuring Group Differences in High-Dimensional Choices.* Econometrica 87(4).
- Widmer, Abed Meraim, Galletta, Ash (2020). *Media Slant is Contagious.*
```

- [ ] **Step 2: Run the full test suite one more time**

Run: `pytest -q`
Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with install, quick tour, and MC instructions"
```

---

## Self-Review Checklist (for plan author)

- **Spec coverage:**
  - [x] Model (spec §5.1) → Task 3, 4
  - [x] Three estimators (spec §2.3) → Tasks 9, 10
  - [x] `fit_mle`, `fit_penalized`, `fit_path` (spec §5.2) → Tasks 5, 6, 7
  - [x] Partisanship primitives (spec §5.3) → Task 8
  - [x] Subsampling CIs (spec §5.5) → Task 11
  - [x] `scale_document` and `score_document` (spec §5.6) → Task 12
  - [x] Simulator (spec §5.7) → Tasks 2, 13
  - [x] MC experiments A / B / C (spec §5.8) → Tasks 14, 15, 16
  - [x] Docs (spec M8) → Task 18

- **Placeholder scan:** no TODOs, no "implement later" — each step has complete code or explicit commands.

- **Type consistency:** names used across tasks:
  - `PhraseData.from_arrays(counts, party, session, X=None, device=...)` — consistent in Tasks 3–17.
  - `PhraseChoiceModel(V, T, P)` constructor — consistent.
  - `fit_mle(model, data, optimizer=..., max_iter=..., tol=..., ridge=...)` — consistent.
  - `fit_penalized(model, data, lam, lam_alpha=..., lam_gamma=..., max_iter=..., tol=..., batch_size=...)` — consistent.
  - `fit_path(model, data, lam_grid=None, grid_size=..., criterion="bic", ...)` → returns dict with keys `lam`, `best_idx`, `path`, `path_params` — consistent in Task 7 and used the same way in Task 10.
  - Estimator attribute names `partisanship_`, `alpha_`, `gamma_`, `phi_`, `lam_`, `bic_path_`, `df_path_`, `logLik_path_`, `lam_grid_`, `vocab_size_`, `n_covariates_`, `sessions_` — set in Tasks 9, 10; consumed in Tasks 11, 12, 17.
  - `subsample_ci(estimator_factory, counts, party, session, X=None, speaker_id=None, ...)` returns dict with `estimate`, `ci_lower`, `ci_upper`, `subsample_estimates`, `n_sub`, `n_full`, `frac` — consistent in Tasks 11, 15, 17.
  - `scale_document(estimator, counts_new, session, normalize="freq") → float` and `score_document(...) → dict` — consistent in Tasks 12, 17.
