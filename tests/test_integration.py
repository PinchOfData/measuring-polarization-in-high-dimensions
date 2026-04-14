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
