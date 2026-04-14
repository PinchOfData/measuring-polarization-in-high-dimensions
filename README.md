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
