import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "motor-owa-v2", "src"))

import numpy as np
import pandas as pd
import pytest

from motor_owa.data import simulate_market
from motor_owa.profiles import all_profiles
from oe4.benchmarks import (equal_weight, min_variance, max_sharpe,
                            mlp_portfolio, anfis_portfolio, _AnfisLite)
from oe4.stability import (ranking_noise_stability, worst_window,
                           profile_sensitivity_matrix)


@pytest.fixture(scope="module")
def px():
    return simulate_market(n_assets=15, n_days=900, seed=11)


def _check_weights(w):
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert (w.values >= -1e-9).all()


def test_equal_weight(px):
    w = equal_weight(px.iloc[100:300].pct_change().dropna())
    _check_weights(w)
    assert np.allclose(w.values, 1 / 15)


def test_min_variance_lower_vol_than_1n(px):
    rets = px.iloc[100:400].pct_change().dropna()
    wmv = min_variance(rets)
    w1n = equal_weight(rets)
    _check_weights(wmv)
    cov = rets.cov().values
    assert wmv.values @ cov @ wmv.values <= w1n.values @ cov @ w1n.values + 1e-12


def test_max_sharpe_valid(px):
    w = max_sharpe(px.iloc[100:400].pct_change().dropna())
    _check_weights(w)
    assert (w.values <= 0.30 + 1e-6).all()


def test_mlp_and_anfis_portfolios(px):
    for fn in (mlp_portfolio, anfis_portfolio):
        w = fn(px, 700, lookback=126, horizon=42, top_n=6)
        _check_weights(w)
        assert len(w) <= 15


def test_anfis_lite_learns_linear():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (300, 4))
    y = 2 * X[:, 0] - X[:, 1] + 0.1
    mdl = _AnfisLite(n_rules=3, seed=0).fit(X, y)
    pred = mdl.predict(X)
    ss = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    assert ss > 0.95  # R2 alto en un problema lineal


def test_noise_stability_decreases(px):
    prof = all_profiles()[2]
    df = ranking_noise_stability(px, 700, prof, n_rep=10)
    assert df["mean_consistency"].iloc[0] >= df["mean_consistency"].iloc[-1]
    assert (df["mean_consistency"] > 0.5).all()


def test_worst_window_bounds(px):
    a, b = worst_window(px)
    assert 0 <= a < b < len(px)


def test_profile_sensitivity_matrix():
    m = profile_sensitivity_matrix()
    assert m.shape == (8, 7)
    assert (m["s=+0"] == 0).all()          # sin sorpresa no hay migracion
    assert (m["s=-3"] <= 0).all()          # perdidas nunca suben el perfil
    assert (m["s=+3"] >= 0).all()
