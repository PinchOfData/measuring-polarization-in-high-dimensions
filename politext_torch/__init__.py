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
