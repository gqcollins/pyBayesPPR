"""Unit tests for the setup/state classes in pyBayesPPR.pyBayesPPR."""

import numpy as np
import pytest

from pyBayesPPR.pyBayesPPR import (
    bpprData,
    bpprPrior,
    bpprSpecs,
    qf_info,
)


def make_prior(**overrides):
    kwargs = dict(
        n_ridge_mean=10.0,
        n_ridge_max=None,
        n_act_max=None,
        df_spline=4,
        prob_relu=2 / 3,
        prior_coefs="zs",
        shape_var_coefs=None,
        scale_var_coefs=None,
        n_dat_min=None,
    )
    kwargs.update(overrides)
    return bpprPrior(**kwargs)


def make_specs(**overrides):
    kwargs = dict(
        n_post=100,
        n_burn=100,
        n_adapt=0,
        n_thin=1,
        w_n_act=None,
        w_feat=None,
        adapt_act_feat=True,
        scale_proj_dir_prop=None,
    )
    kwargs.update(overrides)
    return bpprSpecs(**kwargs)


# --------------------------------------------------------------------------- #
# bpprData
# --------------------------------------------------------------------------- #

def test_feat_type_classification():
    np.random.seed(0)
    n = 100
    X = np.column_stack([
        np.full(n, 3.0),                       # constant   -> ''
        np.random.choice([0.0, 1.0], n),       # binary     -> 'cat'
        np.random.choice([0.0, 1.0, 2.0], n),  # few unique -> 'disc'
        np.random.rand(n),                     # continuous -> 'cont'
    ])
    y = np.random.normal(size=n)
    data = bpprData(X, y)
    prior = make_prior()
    data.summarize(prior)
    assert list(data.feat_type) == ["", "cat", "disc", "cont"]


def test_standardize_zero_mean_unit_sd():
    np.random.seed(0)
    n = 200
    X = np.random.rand(n, 3) * 5 + 2
    y = np.random.normal(size=n)
    data = bpprData(X, y)
    data.summarize(make_prior())
    np.testing.assert_allclose(np.mean(data.X_st, axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.std(data.X_st, axis=0, ddof=1), 1.0, atol=1e-10)


def test_standardize_external_matches_internal():
    np.random.seed(0)
    X = np.random.rand(100, 3)
    y = np.random.normal(size=100)
    data = bpprData(X, y)
    data.summarize(make_prior())
    np.testing.assert_allclose(data.standardize(X), data.X_st)


# --------------------------------------------------------------------------- #
# bpprPrior
# --------------------------------------------------------------------------- #

def test_prior_calibrate_defaults():
    np.random.seed(0)
    X = np.random.rand(200, 4)
    y = np.random.normal(size=200)
    data = bpprData(X, y)
    prior = make_prior()
    data.summarize(prior)
    prior.calibrate(data)
    assert prior.n_dat_min is not None and prior.n_dat_min > prior.df_spline
    assert prior.n_act_max > 0
    assert prior.n_ridge_max > 0
    assert prior.shape_var_coefs == 0.5
    assert prior.scale_var_coefs == data.n / 2


def test_prior_calibrate_warns_small_n_dat_min():
    np.random.seed(0)
    X = np.random.rand(200, 3)
    y = np.random.normal(size=200)
    data = bpprData(X, y)
    prior = make_prior(df_spline=4, n_dat_min=2)
    data.summarize(prior)
    with pytest.warns(UserWarning):
        prior.calibrate(data)
    assert prior.n_dat_min == prior.df_spline + 1


def test_prior_calibrate_all_constant_raises():
    n = 100
    X = np.column_stack([np.full(n, 1.0), np.full(n, 2.0)])
    y = np.random.normal(size=n)
    data = bpprData(X, y)
    prior = make_prior()
    data.summarize(prior)
    with pytest.raises(AssertionError):
        prior.calibrate(data)


def test_prior_rejects_bad_prior_coefs():
    with pytest.raises(AssertionError):
        make_prior(prior_coefs="bogus")


# --------------------------------------------------------------------------- #
# bpprSpecs
# --------------------------------------------------------------------------- #

def test_specs_thinning_truncation():
    specs = make_specs(n_post=105, n_burn=50, n_adapt=10, n_thin=10)
    assert specs.n_post == 100  # 105 - (105 % 10)
    assert specs.n_keep == 10
    assert specs.n_pre == 60
    assert specs.n_draws == 160


def test_specs_thin_gt_post_raises():
    with pytest.raises(AssertionError):
        make_specs(n_post=5, n_thin=10)


def test_specs_scale_proj_dir_prop_range():
    with pytest.raises(AssertionError):
        make_specs(scale_proj_dir_prop=1.5)
    with pytest.raises(AssertionError):
        make_specs(scale_proj_dir_prop=0.0)
    # valid value should not raise
    make_specs(scale_proj_dir_prop=0.5)


def test_specs_calibrate_zeros_constant_feature_weight():
    np.random.seed(0)
    n = 100
    X = np.column_stack([np.full(n, 1.0), np.random.rand(n), np.random.rand(n)])
    y = np.random.normal(size=n)
    data = bpprData(X, y)
    prior = make_prior()
    data.summarize(prior)
    prior.calibrate(data)
    specs = make_specs()
    specs.calibrate(data, prior)
    assert specs.w_feat[0] == 0.0
    assert np.all(specs.w_feat[1:] > 0)


# --------------------------------------------------------------------------- #
# qf_info
# --------------------------------------------------------------------------- #

def test_qf_info_well_conditioned():
    np.random.seed(0)
    B = np.random.rand(50, 3)
    y = np.random.normal(size=50)
    BtB = B.T @ B
    Bty = B.T @ y
    info = qf_info(BtB, Bty)
    assert info.dim == 3
    assert info.chol is not None
    assert info.ls_est is not None
    # qf should equal y'B (B'B)^-1 B'y
    expected_qf = Bty @ np.linalg.solve(BtB, Bty)
    assert info.qf == pytest.approx(expected_qf, rel=1e-8)


def test_qf_info_ill_conditioned_returns_none():
    # Two near-duplicate columns -> huge conditioning ratio -> early return.
    B = np.ones((50, 3))
    B[:, 1] = np.linspace(0, 1, 50)
    B[:, 2] = B[:, 1] + 1e-12
    BtB = B.T @ B
    Bty = B.T @ np.random.normal(size=50)
    info = qf_info(BtB, Bty)
    assert info.dim is None
    assert info.qf is None
