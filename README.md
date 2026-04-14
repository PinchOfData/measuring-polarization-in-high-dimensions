# politext-torch

PyTorch implementation of the Gentzkow–Shapiro–Taddy (2019, *Econometrica*) partisanship estimators, with subsampling confidence intervals and unseen-text scaling after Widmer et al. (2020).

## Install

```bash
conda env create -f environment.yml
conda activate gst
pip install -e ".[dev]"
```

## Which estimator should I use?

- **No covariates?** Use `LeaveOutEstimator`. It's an exact closed-form debiased estimator: fast, unbiased in our Monte Carlo, and has no hyperparameters.
- **Covariates needed** (partialling out state, chamber, demographics, etc.) **or want `φ̂` for scaling unseen texts?** Use `PenalizedEstimator` with default settings (`criterion="cv"`, 5-fold CV, full λ path). CV picks λ much better than BIC for the weak-signal regimes you typically see in text data.
- **`MLEEstimator` is only useful for pedagogy** — to reproduce the upward finite-sample bias in vocabulary size that the politext paper documents. Don't use it for inference.

### Empirical trade-offs at a realistic scale (N=1000, V=10000, T=5)

Single MC replication from `make_mc_A`, m=100 words per speaker, true π ≈ 0.53 per session:

| Estimator                | π̂ bias  | π̂ RMSE | nnz(φ̂)/50000 | λ̂     | Wall (CPU) |
|--------------------------|---------|--------|---------------|--------|------------|
| `LeaveOutEstimator`      | **−0.005** | **0.007**  | —             | —      | **0.2 s**  |
| `MLEEstimator`           | +0.197  | 0.197  | 50000 (dense) | —      | 10 s       |
| `PenalizedEstimator(criterion="bic")` | −0.029  | 0.029  | 1             | 6.24   | 87 s       |
| `PenalizedEstimator()` (default, CV)   | −0.021  | 0.021  | 12 852        | 1.17   | 5.8 min    |

What to read from this:

- **MLE is catastrophically biased at this V** (+20 percentage points above truth). This is the politext paper's Figure 1 headline — don't use MLE.
- **LeaveOut dominates on π̂** — best accuracy at 1000× faster wall-clock than Penalized. If you don't need covariates or φ̂, there's no reason to use anything else.
- **Penalized-BIC over-shrinks** — at weak-signal scales (small per-phrase counts, moderate N), BIC's per-parameter penalty `log(N)·df` is large relative to true φ magnitudes, collapsing φ̂ to zero. That's why CV is the default.
- **Penalized-CV recovers much better** but still pays ~4× more bias than LeaveOut — the price of estimating 50k parameters from sparse counts.
- **Scaling cost**: subsampling CIs multiply these wall-clocks by `n_subsamples` (~100). Penalized-CV inference would be hours on CPU at this scale; LeaveOut inference is ~20 seconds.

### A note on "unbiasedness" and scaling

The paper's unbiasedness claims concern **π̂ (session-level partisanship)**, not φ̂ or document scaling scores.

- `LeaveOutEstimator` is (approximately) unbiased for π̄_t — proven in politext §4.2.
- `PenalizedEstimator`'s `φ̂` is *not* unbiased. L1 always shrinks toward zero, by design.
- `scale_documents(pen, docs, session=t)` computes `Σ_b f_b · φ̂_{b,t}` — a **projection score** (real-valued left/right position), not an estimator of any population quantity. "Unbiased scaling" isn't a meaningful notion here.

Consequence: `LeaveOutEstimator` has no `phi_` attribute and **cannot be used for scaling**. You could back out empirical log-odds `log(q̂^R/q̂^D)` from raw counts, but it re-introduces exactly the finite-sample noise the leave-out trick purges (Jensen on rare phrases → bias explodes as counts get sparse). Not exposed in the library on purpose.

**Practical pattern:** use `LeaveOutEstimator` for π̂ headlines and `PenalizedEstimator` on the same data for `φ̂`-based scaling of unseen texts.

## Quick tour

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from patsy import dmatrix

from politext_torch import (
    LeaveOutEstimator, PenalizedEstimator,
    subsample_ci, scale_documents,
)

# 1. Build counts from text.
vectorizer = CountVectorizer(ngram_range=(2, 2), min_df=5)
counts = vectorizer.fit_transform(list_of_speeches)            # (N, V) sparse
party = np.array(party_list, dtype=float)                      # 0 or 1
session = np.array(session_list, dtype=int)                    # 0..T-1

# 2a. No-covariates path: leave-out estimator.
lo = LeaveOutEstimator().fit(counts, party, session)
#   lo.partisanship_  → (T,) vector of π̂_t

# 2b. With covariates (state, gender, chamber, ...): penalized estimator with CV.
X = np.asarray(dmatrix("state + gender + chamber", data=meta, return_type="matrix"))
pen = PenalizedEstimator().fit(counts, party, session, X)
#   pen.partisanship_   → (T,) vector of π̂_t
#   pen.phi_            → (V, T) partisan phrase loadings
#   pen.lam_            → CV-selected λ

# 3. Subsampling 95% CI.
ci = subsample_ci(
    lambda: LeaveOutEstimator(),          # or PenalizedEstimator() if you need covariates
    counts, party, session,
    n_subsamples=100, frac=0.1, seed=0,
)
# ci["estimate"], ci["ci_lower"], ci["ci_upper"] are all (T,) arrays.

# 4. Scale unseen documents onto the partisan axis (requires PenalizedEstimator).
new_counts = vectorizer.transform(new_docs)                    # REUSE the fitted vectorizer.
scores = scale_documents(pen, new_counts, session=0)           # (M,) vectorized
```

`scale_documents` is a single sparse matvec under the hood — comfortable with millions of documents.

## Running the Monte Carlo experiments

```bash
python -m politext_torch.experiments.mc_bias_rmse    # Exp A: bias/RMSE vs vocabulary size
python -m politext_torch.experiments.mc_coverage     # Exp B: 95% CI coverage
python -m politext_torch.experiments.mc_null         # Exp C: null behaviour under φ=0
```

Figures and CSVs land in `politext_torch/experiments/output/`.

The MC scripts run `PenalizedEstimator` with small `grid_size` / `max_iter` to keep wall-clock under ~15 min each; for real applications use the library defaults (`grid_size=100`, `max_iter=500`).

## References

- Gentzkow, Shapiro, Taddy (2019). *Measuring Group Differences in High-Dimensional Choices.* Econometrica 87(4).
- Widmer, Abed Meraim, Galletta, Ash (2020). *Media Slant is Contagious.*
