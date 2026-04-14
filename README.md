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
