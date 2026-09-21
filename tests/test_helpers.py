"""Unit tests for the pure helper functions in pyBayesPPR.pyBayesPPR."""

import numpy as np
import pytest
from itertools import combinations
from scipy.special import comb

from pyBayesPPR.pyBayesPPR import (
    relu,
    combn,
    lchoose,
    get_cat_basis,
    get_mns_basis,
    rps,
    get_move_type,
    get_log_mh_bd,
    dwallenius,
    autocorrelation,
    effective_sample_size,
    split_chain_into_subchains,
    calculate_rhat,
)


def test_relu_shape_and_clamping():
    x = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])
    out = relu(x)
    assert out.shape == (len(x), 1)
    expected = np.array([0.0, 0.0, 0.0, 1.5, 3.0]).reshape(-1, 1)
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("n,k", [(4, 2), (5, 3), (6, 1), (5, 5)])
def test_combn_matches_itertools(n, k):
    result = combn(n, k)
    expected = np.array(list(combinations(range(n), k)))
    assert result.shape == (comb(n, k, exact=True), k)
    np.testing.assert_array_equal(np.sort(result, axis=0), np.sort(expected, axis=0))


@pytest.mark.parametrize("n,k", [(5, 2), (10, 3), (8, 0), (8, 8)])
def test_lchoose_approximates_log_comb(n, k):
    assert lchoose(n, k) == pytest.approx(np.log(comb(n, k, exact=True)), abs=1e-9)


def test_get_cat_basis_single_column():
    Xj = np.array([[0.0], [1.0], [1.0]])
    np.testing.assert_allclose(get_cat_basis(Xj), Xj)


def test_get_cat_basis_multi_column():
    Xj = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    expected = (1 - np.prod(1 - Xj, axis=1, keepdims=True))
    np.testing.assert_allclose(get_cat_basis(Xj), expected)


def test_get_mns_basis_column_count():
    df_spline = 4
    knot_quants = np.linspace(0.0, 1.0, num=df_spline + 1)
    u = np.linspace(-1.0, 1.0, 50)
    knot0 = np.quantile(u, 0.1)
    knots = np.append([knot0], np.quantile(u[u > knot0], knot_quants))
    basis = get_mns_basis(u, knots)
    assert basis.shape == (len(u), df_spline)


def test_rps_returns_unit_vector():
    np.random.seed(0)
    d = 4
    mu = np.repeat(1 / np.sqrt(d), d)
    x = rps(mu, 0.0)
    assert x.shape == (d,)
    assert np.sqrt(np.sum(x**2)) == pytest.approx(1.0)


def test_get_move_type_boundaries():
    assert get_move_type(0, 0, 10) == "birth"
    assert get_move_type(5, 0, 5) == "death"  # at max, no quant -> death only
    # at max with quant available -> death or change
    np.random.seed(0)
    for _ in range(20):
        assert get_move_type(5, 2, 5) in ("death", "change")
    # interior, no quant -> birth or death
    for _ in range(20):
        assert get_move_type(3, 0, 5) in ("birth", "death")


def test_get_log_mh_bd_branches():
    n_max = 5
    assert get_log_mh_bd(0, 0, n_max) == 0
    assert get_log_mh_bd(n_max, 0, n_max) == 0
    assert get_log_mh_bd(n_max, 1, n_max) == pytest.approx(np.log(2))
    assert get_log_mh_bd(2, 0, n_max) == pytest.approx(np.log(2))
    assert get_log_mh_bd(2, 1, n_max) == pytest.approx(np.log(3))


def test_dwallenius_full_selection_is_one():
    w = np.array([0.2, 0.3, 0.5])
    feat = np.array([0, 1, 2])
    assert dwallenius(w, feat) == 1.0


def test_autocorrelation_lag_zero_is_one():
    np.random.seed(0)
    chain = np.random.normal(size=200)
    assert autocorrelation(chain, 0) == pytest.approx(1.0)


def test_autocorrelation_invalid_lag_raises():
    chain = np.arange(10.0)
    with pytest.raises(ValueError):
        autocorrelation(chain, -1)
    with pytest.raises(ValueError):
        autocorrelation(chain, 10)


def test_autocorrelation_zero_variance():
    chain = np.ones(10)
    assert autocorrelation(chain, 1) == 0


def test_effective_sample_size_short_and_constant():
    assert effective_sample_size(np.array([1.0])) == 1
    assert effective_sample_size(np.ones(50)) == 1


def test_effective_sample_size_bounded_by_n():
    np.random.seed(0)
    chain = np.random.normal(size=500)
    ess = effective_sample_size(chain)
    assert 0 < ess <= len(chain)


def test_split_chain_into_subchains_shape():
    chain = np.arange(103)
    sub = split_chain_into_subchains(chain, 4)
    assert sub.shape == (4, 25)  # 103 trimmed to 100, split into 4


def test_calculate_rhat_degenerate_is_nan():
    assert np.isnan(calculate_rhat(np.ones((4, 25))))  # no within-chain variance
    assert np.isnan(calculate_rhat(np.ones((1, 25))))  # too few chains


def test_calculate_rhat_iid_near_one():
    np.random.seed(0)
    chain = np.random.normal(size=4000)
    rhat = calculate_rhat(split_chain_into_subchains(chain, 4))
    assert rhat == pytest.approx(1.0, abs=0.1)
