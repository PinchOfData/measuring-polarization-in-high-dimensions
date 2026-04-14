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
