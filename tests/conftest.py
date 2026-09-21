"""Shared pytest fixtures and configuration for the pyBayesPPR test suite.

Importing this module forces matplotlib's non-interactive ``Agg`` backend *before*
``pyBayesPPR`` (and therefore ``matplotlib.pyplot``) is imported anywhere, so that
plotting code never tries to open a window during a test run or in CI.
"""

import matplotlib

# Must run before pyBayesPPR imports matplotlib.pyplot.
matplotlib.use("Agg")

import numpy as np
import pytest

from pyBayesPPR import bppr


def friedman(X):
    """The Friedman function -- only the first 5 columns of ``X`` are used."""
    return (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20.0 * (X[:, 2] - 0.5) ** 2
        + 10.0 * X[:, 3]
        + 5.0 * X[:, 4]
    )


@pytest.fixture(autouse=True)
def _seed_rng():
    """Seed the global numpy RNG before every test for determinism.

    The library uses the global ``np.random`` state throughout, so seeding here
    makes both the fitting path and the sampling-based helpers reproducible.
    """
    np.random.seed(0)


def _make_data(n=500, p=10, seed=0):
    rng_state = np.random.get_state()
    np.random.seed(seed)
    X = np.random.rand(n, p)
    y = np.random.normal(friedman(X), 1.0)
    np.random.set_state(rng_state)
    return X, y


@pytest.fixture(scope="session")
def training_data():
    """Friedman training data: (X, y) with n=500, p=10 (5 active features)."""
    return _make_data(n=500, p=10, seed=0)


@pytest.fixture(scope="session")
def test_data():
    """Held-out Friedman data for prediction-accuracy checks."""
    return _make_data(n=400, p=10, seed=1)


@pytest.fixture(scope="session")
def fitted_model(training_data):
    """A single bppr() fit (Zellner-Siow prior) shared across integration tests.

    Uses a small MCMC budget so the whole integration suite runs in a few seconds
    while still exercising the full RJMCMC birth/death/change path.
    """
    X, y = training_data
    np.random.seed(0)
    return bppr(X, y, n_post=200, n_burn=200, n_thin=1, silent=True)
