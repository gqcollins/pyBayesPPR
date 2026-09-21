"""Headless smoke test for the end-to-end pyBayesPPR workflow.

This replaces the former interactive demo (which called ``plt.show()``). It runs
the full fit -> predict -> sobol -> plot pipeline with matplotlib's non-interactive
``Agg`` backend so it can execute under ``pytest`` and in CI without opening windows.

The richer assertions live in the top-level ``tests/`` package; this module keeps a
minimal, self-contained smoke test alongside the source.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from pyBayesPPR import bppr


def f(X):  # Friedman function
    return (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20.0 * (X[:, 2] - 0.5) ** 2
        + 10.0 * X[:, 3]
        + 5.0 * X[:, 4]
    )


def test_end_to_end(tmp_path):
    np.random.seed(0)

    # Generate data.
    n = 500  # sample size
    p = 10  # number of predictors (only 5 are used)
    X = np.random.rand(n, p)  # predictors (training set)
    y = np.random.normal(f(X), 1)  # response (training set) with noise.

    # Fit BayesPPR model with RJMCMC (small budget so the test is fast).
    mod = bppr(X, y, n_post=200, n_burn=200, n_thin=1, silent=True)

    mod.plot(file=str(tmp_path / "plot.png"))
    mod.traceplot(file=str(tmp_path / "traceplot.png"))
    mod.sobol(mcmc_use="last", n_mc=2**8)
    mod.plot_sobol(file=str(tmp_path / "sobol.png"))

    # Predict at new inputs and check the prediction shape/accuracy.
    X_test = np.random.rand(1000, p)
    y_test = np.random.normal(f(X_test), 1)

    pred = mod.predict(X_test)
    assert pred.shape == (mod.specs.n_keep, X_test.shape[0])

    post_mean = np.mean(pred, axis=0)
    r_squared = 1 - np.mean((y_test - post_mean) ** 2) / np.var(y_test)
    assert r_squared > 0.8

    mod.plot(X_test, y_test, file=str(tmp_path / "plot_test.png"))
