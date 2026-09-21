"""Integration tests: fit bppr() and validate predict / sobol / plotting."""

import numpy as np
import pytest

from pyBayesPPR import bppr, bpprModel


def r_squared(y, pred_mean):
    resid = y - pred_mean
    return 1 - np.mean(resid**2) / np.var(y)


def test_fit_returns_populated_model(fitted_model):
    assert isinstance(fitted_model, bpprModel)
    assert len(fitted_model.samples.n_ridge) == fitted_model.specs.n_keep
    # The Friedman signal should induce at least some ridge functions.
    assert np.max(fitted_model.samples.n_ridge) > 0


def test_predict_shape_and_accuracy(fitted_model, test_data):
    X_test, y_test = test_data
    preds = fitted_model.predict(X_test)
    assert preds.shape == (fitted_model.specs.n_keep, X_test.shape[0])
    post_mean = np.mean(preds, axis=0)
    assert r_squared(y_test, post_mean) > 0.85


def test_predict_mcmc_use_variants(fitted_model, test_data):
    X_test, _ = test_data
    n_keep = fitted_model.specs.n_keep

    single = fitted_model.predict(X_test, mcmc_use=0)
    assert single.shape == (1, X_test.shape[0])

    subset = fitted_model.predict(X_test, mcmc_use=np.array([0, 1, 2]))
    assert subset.shape == (3, X_test.shape[0])

    with pytest.raises(AssertionError):
        fitted_model.predict(X_test, mcmc_use=n_keep)  # out of range


def test_predict_wrong_ncols_raises(fitted_model, test_data):
    X_test, _ = test_data
    with pytest.raises(AssertionError):
        fitted_model.predict(X_test[:, :3])


def test_sobol_shape_and_active_features(fitted_model):
    fitted_model.sobol(mcmc_use="last", n_mc=2**8)
    p = fitted_model.data.p
    assert fitted_model.first_order_sobol.shape == (1, p)
    assert fitted_model.total_order_sobol.shape == (1, p)

    total = np.mean(fitted_model.total_order_sobol, axis=0)
    active = total[:5].mean()      # Friedman uses columns 0-4
    inactive = total[5:].mean()    # columns 5-9 are noise
    assert active > inactive


def test_plots_run_headless(fitted_model, test_data, tmp_path):
    X_test, y_test = test_data
    fitted_model.plot(file=str(tmp_path / "diag.png"))
    fitted_model.plot(X_test, y_test, file=str(tmp_path / "diag_test.png"))
    fitted_model.traceplot(file=str(tmp_path / "trace.png"))
    fitted_model.sobol(mcmc_use="last", n_mc=2**8)
    fitted_model.plot_sobol(file=str(tmp_path / "sobol.png"))
    for name in ("diag.png", "diag_test.png", "trace.png", "sobol.png"):
        assert (tmp_path / name).exists()


def test_flat_prior_fit(training_data, test_data):
    X, y = training_data
    np.random.seed(0)
    model = bppr(X, y, prior_coefs="flat", n_post=200, n_adapt=200,
                 n_burn=0, n_thin=1, silent=True)
    assert model.samples.var_coefs is None  # not sampled under flat prior
    X_test, y_test = test_data
    post_mean = np.mean(model.predict(X_test), axis=0)
    assert r_squared(y_test, post_mean) > 0.85
