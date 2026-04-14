# PyTorch implementation of Gentzkow–Shapiro–Taddy (2019) partisanship estimators

**Status:** design approved (pending spec review)
**Date:** 2026-04-14
**References:**
- *politext*: Gentzkow, Shapiro, Taddy (2019), "Measuring Group Differences in High-Dimensional Choices: Method and Application to Congressional Speech," *Econometrica* 87(4).
- *media_slant*: Widmer, Abed Meraim, Galletta, Ash (2020), "Media Slant is Contagious."

## 1. Goal

A PyTorch library that implements the three estimators of partisanship in Gentzkow–Shapiro–Taddy (2019) — naive MLE plug-in, leave-out, and penalized — together with subsampling-based confidence intervals, a method to project unseen documents onto the learned partisan axis (per Widmer et al.), and Monte Carlo experiments that verify the estimators behave as the theory predicts.

## 2. Scope

### 2.1 Model (politext §3)

Observed counts: for each (speaker, session) row `i`, the phrase-count vector `c_i ∈ ℕ^V` with verbosity `m_i = Σ_j c_ij`.

Generative model (multinomial logit):

```
c_i ~ MN(m_i, q_i)
q_ij = exp(u_ij) / Σ_l exp(u_il)
u_ij = α_{j,t_i} + x_i' γ_j + φ_{j,t_i} · K_i
```

Parameters:

- `α ∈ ℝ^{V×T}` — phrase × session baseline log-rate
- `γ ∈ ℝ^{V×P}` — phrase × covariate loading (shared across sessions)
- `φ ∈ ℝ^{V×T}` — phrase × session partisan loading

`K_i ∈ {0, 1}` is the Republican indicator. `x_i ∈ ℝ^P` is a patsy-built covariate vector; session-varying covariates are handled by interacting them with session in the patsy formula (e.g., `C(census_region):C(session)`), which produces flat columns of `X` absorbed by `γ`.

Estimation uses the **Poisson approximation** from politext §4.3, Taddy (2015):

```
c_ij ~ Pois(m_i · exp(u_ij))
```

This makes the likelihood separable across phrases and enables distributed/vectorized computation.

### 2.2 Partisanship target (politext §3.2)

Per-speaker partisanship at characteristics `x`:

```
π_t(x) = ½ · Σ_j q_{jt}^R(x) · ρ_{jt}(x) + ½ · Σ_j q_{jt}^D(x) · (1 − ρ_{jt}(x))
ρ_{jt}(x) = q_{jt}^R(x) / (q_{jt}^R(x) + q_{jt}^D(x))
```

Session-average partisanship:

```
π̄_t = (1 / |R_t ∪ D_t|) · Σ_{i ∈ R_t ∪ D_t} π_t(x_i)
```

`π̄_t ∈ [0.5, 1]` is the primary output.

### 2.3 Three estimators

| Estimator | How π̄_t is computed | Covariates | Implementation path |
| --- | --- | --- | --- |
| **MLE plug-in** (politext §4.1) | Empirical `q̂^P` and `q̂^P/(q̂^R+q̂^D)` plugged into eq. (3); equivalent to using fitted probs from unpenalized fit. | Supported via the unpenalized Poisson fit of eq. (9). | `fit_mle` then `partisanship(...)` |
| **Leave-out** (politext §4.2, eq. 8) | Uses own speaker's `q̂_i` paired with leave-one-speaker-out `ρ̂_{-i,t}`. | Not supported (paper §4.2 is explicit about this limitation). | Closed-form from counts; no model fit. |
| **Penalized** (politext §4.3, eq. 9) | Fitted probs from Poisson + L1 on `φ` + small ridge on `α`, `γ`. | Supported. | `fit_penalized` or `fit_path` + BIC. |

### 2.4 Scaling unseen text (media_slant §3.3)

Two projection methods, both on a `PenalizedEstimator`:

1. **Dot-product projection** (media_slant eq. 1):
   `score(doc) = Σ_b f_b · φ̂_{b,t}` where `f_b` is the relative frequency of phrase `b` in the document.
2. **Model-based posterior** (politext eq. 3–4 applied to the new doc treated as a hypothetical speaker):
   Compute `q^R`, `q^D`, `ρ` at the doc's covariates `x_new`, then the posterior `π` the neutral-prior observer would assign.

### 2.5 Monte Carlo experiments

Three scripts validating that the estimators behave as theory predicts:

- **A.** Bias/RMSE as vocabulary size V grows — MLE biased upward; leave-out and penalized track truth.
- **B.** 95% CI coverage via subsampling for leave-out and penalized.
- **C.** Null behaviour under `φ = 0` (no true partisanship) — MLE biased above 0.5; leave-out and penalized centered on 0.5.

## 3. Non-goals

- **Full-multinomial likelihood.** We use the Poisson approximation exclusively (matches paper, enables per-phrase parallelism).
- **Per-session independent fits.** The default is joint multi-session fit with `γ` shared; no per-session-only mode.
- **Text preprocessing.** The library consumes `scipy.sparse.csr_matrix` count outputs from `sklearn.feature_extraction.text.CountVectorizer` and patsy-built covariate matrices. Tokenization, stemming, and stopword removal are upstream of this library.
- **Parametric bootstrap** and **cross-validation-based λ selection beyond K-fold**. The MC coverage experiment exercises the subsampling CIs directly.
- **Non-speaker-level subsampling.** Subsampling is always at the `speaker_id` level (default: one speaker per row).

## 4. Dependencies

| Package | Purpose |
| --- | --- |
| `torch` (≥ 2.0) | Core tensors, autograd, sparse COO, L-BFGS, Adam. |
| `numpy` | Array interop at boundaries. |
| `scipy.sparse` | Input format for counts (CSR). |
| `scikit-learn` | User-facing; we consume its `CountVectorizer` output. Not a hard import inside the library. |
| `patsy` | User-facing; we consume its design-matrix output. Not a hard import inside the library. |
| `joblib` | Optional, for parallel subsampling. |
| `pytest` | Tests. |
| `matplotlib` | MC experiment figures. |

The library itself imports only `torch`, `numpy`, and `scipy.sparse` to keep the core slim; sklearn and patsy are not hard-required at import time.

## 5. Architecture

Layered design:

```
model.py         PhraseChoiceModel (nn.Module): parameters + Poisson NLL
fit.py           fit_mle, fit_penalized (FISTA), fit_path (BIC/CV)
partisanship.py  choice_probs, posterior_rho, partisanship,
                 leave_out_rho, leave_out_partisanship
estimators.py    MLEEstimator, LeaveOutEstimator, PenalizedEstimator
inference.py     subsample_ci
scale.py         scale_document, score_document (+ batched variants)
simulate.py      draw_counts(params, X, party, session, m)
```

`experiments/` holds the three MC scripts. `tests/` has one file per module.

### 5.1 `model.py` — `PhraseChoiceModel`

`nn.Module` holding parameters `alpha (V,T)`, `gamma (V,P)`, `phi (V,T)` as `torch.nn.Parameter`.

Inputs to `forward` (packaged in a `PhraseData` dataclass):

- `counts`: `torch.sparse_coo_tensor` of shape `(N, V)` built once from the CSR input.
- `log_m`: `(N,)` dense, precomputed offset.
- `party`: `(N,)` dense float in {0, 1}.
- `session`: `(N,)` dense long in `0..T-1`.
- `X`: `(N, P)` dense float.

Poisson NLL is computed in two parts:

1. **Rate sum** `Σ_{i,j} m_i · exp(u_ij)` — iterated in **document-batches** of user-tunable size `batch_size` (default 512). For each batch `B`:

   ```
   u_B = alpha[:, session[B]].T + X[B] @ gamma.T + phi[:, session[B]].T * party[B, None]
   rate_sum += (log_m[B, None] + u_B).exp().sum()
   ```

   Memory per batch is `O(|B| · V)`.

2. **Data-fit term** `Σ_{(i,j): c_ij>0} c_ij · u_ij` — computed on the sparse nnz entries:

   ```
   idx = counts.coalesce().indices()   # (2, nnz)
   vals = counts.coalesce().values()   # (nnz,)
   i, j = idx[0], idx[1]
   u_obs = alpha[j, session[i]] + (X[i] * gamma[j]).sum(-1) + phi[j, session[i]] * party[i]
   data_fit += (vals * u_obs).sum()
   ```

   Memory is `O(nnz)`.

`loss = rate_sum - data_fit + (ψ_α/2) ‖α‖² + (ψ_γ/2) ‖γ‖²`.
L1 on `φ` is *not* in the forward — applied by the fitter's proximal step.

**Initialization:** `alpha[:, t] = log((Σ_{i ∈ t} c_ij + ε) / (Σ_{i ∈ t} m_i + ε))`; `gamma = 0`; `phi = 0`.

**Device:** `model.to(device)` moves all params and cached data tensors. Sparse COO supports CUDA.

### 5.2 `fit.py`

**`fit_mle(model, data, optimizer="lbfgs", max_iter=100, tol=1e-6, ridge=1e-5, verbose=False)`**

- Unpenalized. Poisson NLL is convex; L-BFGS default. Adam as alternative via `optimizer="adam"` with extra kwargs `lr`, `batch_fraction` (for stochastic accumulation across doc-batches).
- Stops on relative loss change `< tol` or iteration cap.

**`fit_penalized(model, data, lam, lam_alpha=1e-5, lam_gamma=1e-5, max_iter=500, tol=1e-5, backtracking=True, verbose=False)`**

- FISTA with Nesterov extrapolation on all params; soft-threshold prox applied to `φ` only. L2 on `α, γ` is inside the smooth part.
- Backtracking line search on step `η`, halved until majorization holds; step size reused as warm start at the next call.

**`fit_path(model, data, lam_grid=None, criterion="bic", cv_folds=5, **kwargs) → {"lam": float, "path": list, "best_idx": int, "model": PhraseChoiceModel}`**

- Default grid: log-spaced `G=100` values from `λ_max` down to `λ_max / 1000`. `λ_max = max_{j,t} |∂NLL/∂φ_jt|` at `φ = 0` (analytic lasso starting point).
- Warm-start: each solution initializes the next.
- `criterion="bic"` (default): `BIC = -2 · logLik + log(n) · df`, `df` = number of nonzero `φ` entries; `n` = `N`.
- `criterion="cv"`: K-fold out-of-sample Poisson deviance; K-fold at the speaker level.
- Returns a dict with per-λ diagnostics (logLik, BIC, df, #iters) and the fitted model at the selected λ.

### 5.3 `partisanship.py`

Pure functions; no state. All accept fitted tensors and return tensors.

- `choice_probs(alpha_t, gamma, phi_t, X_row, party) → (V,)` — softmax over phrases.
- `posterior_rho(alpha_t, gamma, phi_t, X_row) → (V,)` — ρ_jt(x).
- `partisanship(alpha, gamma, phi, X, session, party) → (T,)` — eq. (3)–(5), vectorized across phrases, looped over sessions.
- `leave_out_rho(counts, party, session, speaker_id) → sparse (nnz,)-indexed tensor` — per-speaker leave-one-out ρ̂ values at the phrases they actually used.
- `leave_out_partisanship(counts, party, session, speaker_id) → (T,)` — eq. (8), computed from scatter-add'd group sums.

### 5.4 `estimators.py`

Common base class `BasePartisanshipEstimator` with:

- `fit(counts, party, session, X=None, speaker_id=None, **fit_kwargs) → self`
- `partisanship_`: `numpy.ndarray` of shape `(T,)`
- `sessions_`, `vocab_size_`, `n_covariates_`: fitted metadata
- `to(device)`

**`MLEEstimator(optimizer="lbfgs", max_iter=100, tol=1e-6, ridge=0.0, device="cpu")`** — runs `fit_mle`, then calls `partisanship(...)`. Default `ridge=0.0` so the fit reproduces the paper's empirical plug-in (eq. 6) up to optimizer tolerance when `X=None`; users can set a small positive `ridge` to stabilize pathological sessions. Stores `alpha_`, `gamma_`, `phi_` as numpy.

**`LeaveOutEstimator`** — calls `leave_out_partisanship(...)` only. Ignores `X` with a `UserWarning` if provided. No fitted model parameters.

**`PenalizedEstimator(lam=None, lam_grid=None, criterion="bic", cv_folds=5, store_path=False, lam_alpha=1e-5, lam_gamma=1e-5, max_iter=500, tol=1e-5, device="cpu")`** — dispatches between `fit_penalized` (single λ) and `fit_path` (grid). Exposes:

- `lam_`, `lam_grid_`, `bic_path_`, `df_path_`, `logLik_path_`
- `alpha_`, `gamma_`, `phi_` (numpy)
- `path_models_` is *not* retained (memory); instead `path_params_` stores param snapshots as compact numpy arrays only if `store_path=True`.

**Edge cases, identical across all three:**

- Sessions with 0 Republicans or 0 Democrats: `partisanship_[t] = nan`, `UserWarning` emitted once.
- Phrases with zero count in a session: handled by the ridge + the ε in init.
- `speaker_id` defaults to `np.arange(N)`.

### 5.5 `inference.py`

**`subsample_ci(estimator_factory, counts, party, session, X=None, speaker_id=None, n_subsamples=100, frac=0.1, alpha=0.05, transform="log", seed=None, n_jobs=1, device="cpu") → dict`**

Procedure:

1. Full-sample fit from `estimator_factory()`; store as `π̂`.
2. Draw `n_subsamples` subsamples of unique `speaker_id`s *without replacement*, each of size `round(frac · n_speakers)`.
3. Fit a fresh estimator per subsample; collect `π̂^{(b)}_t`.
4. CI construction (per session `t`, following Politis–Romano–Wolf 1999 Thm 2.2.1):
   - `transform="identity"`: let `Q^{(b)}_t = √τ_b · (π̂^{(b)}_t − π̂_t)`, form CI as `π̂_t − q_{1-α/2}(Q) / √n_full`, `π̂_t − q_{α/2}(Q) / √n_full`.
   - `transform="log"` (default, paper): same construction on `g(π) = log(π − ½)`, i.e. compute `Q^{(b)}_t = √τ_b · (g(π̂^{(b)}_t) − g(π̂_t))`, form CI on `g` scale, then map back with `g⁻¹(u) = ½ + exp(u)`. This is the exact form in politext Figure 1 notes. Falls back to `"identity"` on a per-session basis when `π̂_t − ½ < 1e-6`.
5. Returns:
   ```
   {
     "estimate": (T,) np.ndarray,
     "ci_lower": (T,) np.ndarray,
     "ci_upper": (T,) np.ndarray,
     "subsample_estimates": (n_subsamples, T) np.ndarray,
     "n_sub": int,
     "n_full": int,
     "frac": float,
   }
   ```

**Parallelism:** `joblib.Parallel(n_jobs=n_jobs)` over subsamples. On GPU, default `n_jobs=1`; on CPU users can raise it.

**Reproducibility:** `seed` seeds both numpy (subsample selection) and each worker's torch RNG deterministically.

### 5.6 `scale.py`

`scale_document(estimator, counts_new, session, normalize="freq") → float`
`scale_documents(estimator, counts_matrix, session) → (M,)`
`score_document(estimator, counts_new, session, X_new=None) → {"pi": float, "rho": (V,), "q_R": (V,), "q_D": (V,)}`
`score_documents(estimator, counts_matrix, session, X_new=None) → dict of arrays`

**Vocabulary alignment (load-bearing):** these functions require that `counts_new` comes from the *same fitted* `CountVectorizer` used at training. Runtime guard: `counts_new.shape[-1] == estimator.vocab_size_`, else `ValueError` with explicit guidance ("use vectorizer.transform(), do not refit"). Docstrings and README call out the same requirement.

**`normalize` options:**

- `"freq"` (default, paper): relative frequency.
- `"count"`: raw counts.
- `"binary"`: phrase indicator (useful for short snippets, per media_slant fn. 6).

### 5.7 `simulate.py`

`draw_counts(alpha, gamma, phi, X, party, session, m, seed=None) → scipy.sparse.csr_matrix`

Computes `u`, softmaxes to choice probs, samples `c_i ~ Multinomial(m_i, q_i)` (via `torch.distributions.Multinomial` or numpy's `multinomial`), returns sparse CSR to stay consistent with the library's input format.

Also provides builders for canonical MC setups: `make_mc_A(V, N, T, seed)`, `make_mc_B(...)`, `make_mc_C(...)`, returning tuples `(counts, party, session, X, true_pi)` used by the experiments and by tests.

### 5.8 `experiments/`

Three scripts, each runnable as `python -m politext_torch.experiments.<name>`:

- `mc_bias_rmse.py` — sweep V, 200 reps each, plot bias/RMSE vs V for all three estimators. Saves `fig_a.pdf`, `results_a.csv`.
- `mc_coverage.py` — 500 reps at fixed DGP, compute 95% CI coverage for leave-out and penalized. Saves `fig_b.pdf`, `results_b.csv`.
- `mc_null.py` — `φ=0`, 200 reps, plot distribution of π̂ per estimator. Saves `fig_c.pdf`, `results_c.csv`.

Top-of-file constants for sizes and `n_rep`; designed to run in a few minutes on CPU at default sizes and scale up cleanly on GPU.

## 6. Tests

Per-module unit tests:

- **`test_model.py`** — forward NLL matches a numpy reference on a tiny (N=5, V=4, T=2, P=1) case; gradient check via `torch.autograd.gradcheck`; batched rate sum matches un-batched.
- **`test_fit.py`** — `fit_mle` recovers true params on a large-m, small-V sample (noise negligible); `fit_penalized` sets `φ` to zero at very large `λ`; warm-start path monotonicity in `λ` ordering.
- **`test_partisanship.py`** — `partisanship` matches a hand-computed 3-phrase 2-party example; `leave_out_partisanship` matches a direct (inefficient) Python reference on a 10-speaker example.
- **`test_estimators.py`** — `fit` populates all declared `*_` attributes; re-fitting is idempotent; `nan` on zero-R or zero-D sessions with warning.
- **`test_inference.py`** — subsampling returns correct shapes; CI contains the point estimate; seed reproducibility.
- **`test_scale.py`** — dot-product score equals manual computation on a 2-phrase example; vocab-mismatch raises; `X_new=None` path works.
- **`test_simulate.py`** — empirical frequencies converge to theoretical softmax probs as `m → ∞`.

Total target: ≈ 50 tests, mostly sub-second.

## 7. Milestones

1. **M1 — Model + simulation + unit tests.** `PhraseChoiceModel`, `draw_counts`, tests for both. End state: can simulate and score a tiny DGP.
2. **M2 — Fitters.** `fit_mle`, `fit_penalized`, `fit_path` + tests. End state: can recover params from a simulation.
3. **M3 — Partisanship + leave-out.** Pure functions + tests. End state: `partisanship(...)` and `leave_out_partisanship(...)` verified against hand-computed cases.
4. **M4 — Estimators.** Three sklearn-style wrappers + tests. End state: end-to-end `fit` on simulated data for all three.
5. **M5 — Inference.** `subsample_ci` + tests. End state: CIs on simulated data with plausible width.
6. **M6 — Scaling.** `scale_document`, `score_document` + tests. End state: fitted `PenalizedEstimator` projects a new doc.
7. **M7 — Monte Carlo experiments.** Three scripts + their expected plots. End state: figures A/B/C reproducing the paper's qualitative results.
8. **M8 — Docs.** README with a short tutorial (simulate → fit → inference → scale). No API docs beyond docstrings for v1.

Each milestone has a commit gate: tests pass, the changed module and its direct tests are reviewed before moving on.

## 8. Open issues / risks

- **Memory at politext-scale vocab (V ≈ 500k).** The document-batched rate sum still requires `O(|B| · V)`. At `|B|=32, V=500k` that's ~64 MB/batch, fine; FISTA gradient accumulation may need `|B|=8`. Users can tune.
- **Sparse-COO on CUDA.** Some PyTorch sparse ops lag behind dense ops. If we hit a missing op, fall back to CSR via `torch.sparse_csr_tensor` or densify the affected operation.
- **L-BFGS with stochastic rate-sum accumulation.** L-BFGS assumes a deterministic loss in the closure. We must always iterate over *all* doc-batches per closure call, not subsample — document this clearly.
- **BIC `df`.** Paper counts nonzero `φ` entries; it omits the `α` degrees of freedom from the BIC penalty per footnote 15. We follow that convention; note in code.
- **Log-transform CI boundary.** At sessions where `π̂_t` is close to 0.5, the log transform becomes unstable. We fall back to the identity transform with a warning when `π̂_t − 0.5 < 1e-6`.

## 9. Out-of-scope extensions (future)

- Joint multi-session fit with a random-effects speaker term (politext footnote: "specification with unobserved speaker characteristics").
- Full multinomial likelihood path for small V.
- Alternative divergence measures (Euclidean, mutual information) — paper supplement mentions them.
- Bayesian inference via variational or MCMC.
