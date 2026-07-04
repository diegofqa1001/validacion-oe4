"""Comparadores del anteproyecto (Obj. 4): media-varianza, ANFIS, red neuronal.

Todos consumen la MISMA ventana de precios que el motor (sin look-ahead) y
devuelven pesos de cartera sobre el universo, para comparacion justa.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

_ANNUAL = 252

__all__ = ["equal_weight", "min_variance", "max_sharpe", "mlp_portfolio",
           "anfis_portfolio", "_AnfisLite", "_TinyMLP"]


def _cap_norm(w: np.ndarray, cap: float = 0.30) -> np.ndarray:
    w = np.clip(np.asarray(w, float), 0, None)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / w.size
    for _ in range(50):
        over = w > cap
        if not over.any():
            break
        exc = (w[over] - cap).sum(); w[over] = cap
        under = ~over
        if w[under].sum() > 0:
            w[under] += exc * w[under] / w[under].sum()
        else:
            w += exc / w.size
    return w / w.sum()




def _active_set_solution(S: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solucion largo-solo de w ~ S^{-1} b via conjunto activo (Markowitz).

    Resuelve iterativamente en el soporte: los activos con peso negativo
    salen del soporte hasta que la solucion es no negativa (condiciones
    KKT del problema de minima varianza / tangencia sin cortos).
    """
    n = S.shape[0]
    active = list(range(n))
    ridge = 1e-8 * np.trace(S) / n
    for _ in range(n):
        Sa = S[np.ix_(active, active)] + ridge * np.eye(len(active))
        try:
            wa = np.linalg.solve(Sa, b[active])
        except np.linalg.LinAlgError:
            wa = np.ones(len(active))
        if (wa >= -1e-12).all():
            w = np.zeros(n)
            w[active] = np.maximum(wa, 0.0)
            s = w.sum()
            return w / s if s > 0 else np.ones(n) / n
        active = [a for a, v in zip(active, wa) if v > 0]
        if not active:
            return np.ones(n) / n
    w = np.zeros(n); w[active] = 1.0 / len(active)
    return w

def equal_weight(rets: pd.DataFrame) -> pd.Series:
    """Cartera 1/N (benchmark ingenuo de DeMiguel et al.)."""
    n = rets.shape[1]
    return pd.Series(np.ones(n) / n, index=rets.columns)


def min_variance(rets: pd.DataFrame, cap: float = 0.30) -> pd.Series:
    """Media-varianza: minima varianza global con largo-solo y tope.

    min w' S w  s.a.  w >= 0, sum w = 1, w <= cap.
    Gradiente proyectado (numpy puro; usa scipy.SLSQP si esta instalado).
    """
    S = rets.cov().values
    n = S.shape[0]
    try:
        from scipy.optimize import minimize
        res = minimize(lambda w: w @ S @ w, np.ones(n) / n, method="SLSQP",
                       bounds=[(0.0, cap)] * n,
                       constraints=[{"type": "eq",
                                     "fun": lambda w: w.sum() - 1}],
                       options={"maxiter": 300})
        return pd.Series(_cap_norm(res.x, cap), index=rets.columns)
    except ImportError:
        w = _active_set_solution(S, np.ones(n))
        return pd.Series(_cap_norm(w, cap), index=rets.columns)


def max_sharpe(rets: pd.DataFrame, cap: float = 0.30,
               rf: float = 0.0) -> pd.Series:
    """Media-varianza: maximo Sharpe ex-ante largo-solo con tope (Markowitz).

    Gradiente proyectado sobre -Sharpe (numpy puro; scipy si existe).
    """
    mu = rets.mean().values * _ANNUAL - rf
    S = rets.cov().values * _ANNUAL
    n = len(mu)

    def neg_sharpe(w):
        v = np.sqrt(max(w @ S @ w, 1e-12))
        return -(w @ mu) / v
    try:
        from scipy.optimize import minimize
        res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                       bounds=[(0.0, cap)] * n,
                       constraints=[{"type": "eq",
                                     "fun": lambda w: w.sum() - 1}],
                       options={"maxiter": 300})
        return pd.Series(_cap_norm(res.x, cap), index=rets.columns)
    except ImportError:
        # cartera tangente largo-solo: w ~ S^{-1} mu con conjunto activo
        w = _active_set_solution(S, np.maximum(mu, 0.0))
        if w.sum() <= 0:
            w = np.ones(n) / n
        return pd.Series(_cap_norm(w, cap), index=rets.columns)


# ---------------- features supervisadas (para MLP y ANFIS) ----------------
def _features_and_target(prices: pd.DataFrame, t: int, lookback: int,
                         horizon: int, n_samples: int = 8):
    """Muestras (X: 4 criterios en t-k; y: retorno del horizonte siguiente).

    Entrena SOLO con informacion previa a t (sin fuga): las muestras se
    toman en t-h, t-2h, ..., y el objetivo es el retorno realizado de ese
    horizonte, que termina como maximo en t.
    """
    from motor_owa.criteria import compute_criteria
    X, y = [], []
    for k in range(1, n_samples + 1):
        tk = t - k * horizon
        if tk - lookback < 1:
            break
        crit = compute_criteria(prices, tk, lookback)
        fwd = prices.iloc[min(tk + horizon, t)] / prices.iloc[tk] - 1.0
        for a in crit.index:
            X.append(crit.loc[a].values)
            y.append(float(fwd[a]))
    return np.array(X), np.array(y)


def mlp_portfolio(prices: pd.DataFrame, t: int, lookback: int = 126,
                  horizon: int = 63, top_n: int = 10, cap: float = 0.30,
                  seed: int = 0) -> pd.Series:
    """Red neuronal (MLP) que predice el retorno del proximo horizonte.

    Cartera: top_n activos por prediccion, pesos inverso-volatilidad.
    """
    from motor_owa.criteria import compute_criteria
    X, y = _features_and_target(prices, t, lookback, horizon)
    crit = compute_criteria(prices, t, lookback)
    if len(y) < 30:  # historia insuficiente: degrada a 1/N
        return equal_weight(prices.iloc[t - lookback:t].pct_change().dropna())
    try:
        from sklearn.neural_network import MLPRegressor
        mdl = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=800,
                           random_state=seed, early_stopping=False)
        mdl.fit(X, y)
        pred_v = mdl.predict(crit.values)
    except ImportError:
        mdl = _TinyMLP(n_hidden=16, seed=seed).fit(X, y)
        pred_v = mdl.predict(crit.values)
    pred = pd.Series(pred_v, index=crit.index)
    sel = pred.nlargest(top_n).index
    rets = prices[sel].iloc[t - lookback:t].pct_change().dropna()
    iv = 1.0 / np.maximum(rets.std().values, 1e-9)
    return pd.Series(_cap_norm(iv, cap), index=sel)




class _TinyMLP:
    """Perceptron multicapa minimo (numpy): 1 capa oculta tanh + salida
    lineal, descenso de gradiente con estandarizacion de entradas.
    Comparador de red neuronal reproducible sin dependencias.
    """

    def __init__(self, n_hidden: int = 16, lr: float = 0.05,
                 epochs: int = 600, seed: int = 0):
        self.h, self.lr, self.epochs, self.seed = n_hidden, lr, epochs, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu_) / self.sd_
        n, d = Xs.shape
        self.W1 = rng.normal(0, 1 / np.sqrt(d), (d, self.h))
        self.b1 = np.zeros(self.h)
        self.W2 = rng.normal(0, 1 / np.sqrt(self.h), self.h)
        self.b2 = 0.0
        for _ in range(self.epochs):
            H = np.tanh(Xs @ self.W1 + self.b1)
            out = H @ self.W2 + self.b2
            err = out - y
            gW2 = H.T @ err / n
            gb2 = err.mean()
            dH = np.outer(err, self.W2) * (1 - H ** 2)
            gW1 = Xs.T @ dH / n
            gb1 = dH.mean(0)
            self.W2 -= self.lr * gW2; self.b2 -= self.lr * gb2
            self.W1 -= self.lr * gW1; self.b1 -= self.lr * gb1
        return self

    def predict(self, X):
        Xs = (X - self.mu_) / self.sd_
        return np.tanh(Xs @ self.W1 + self.b1) @ self.W2 + self.b2

# ---------------- ANFIS ligero (Takagi-Sugeno de primer orden) -------------
class _AnfisLite:
    """ANFIS de primer orden sin dependencias: reglas por k-means +
    pertenencias gaussianas + consecuentes lineales por minimos cuadrados
    ponderados. Equivale a un Takagi-Sugeno entrenado en un solo paso
    (sin backprop), suficiente y reproducible como comparador.
    """

    def __init__(self, n_rules: int = 4, seed: int = 0):
        self.n_rules = n_rules
        self.seed = seed

    def _kmeans(self, X, iters=50):
        rng = np.random.default_rng(self.seed)
        C = X[rng.choice(len(X), self.n_rules, replace=False)]
        for _ in range(iters):
            d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
            lab = d.argmin(1)
            for j in range(self.n_rules):
                if (lab == j).any():
                    C[j] = X[lab == j].mean(0)
        return C, lab

    def fit(self, X, y):
        C, lab = self._kmeans(X)
        self.centers = C
        self.sigmas = np.array([X[lab == j].std(0).mean() + 1e-3
                                for j in range(self.n_rules)])
        W = self._memberships(X)                     # [N, R]
        Xa = np.hstack([X, np.ones((len(X), 1))])    # afin
        self.coefs = []
        for j in range(self.n_rules):
            sw = np.sqrt(W[:, j] + 1e-9)
            A = Xa * sw[:, None]
            b = y * sw
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            self.coefs.append(coef)
        self.coefs = np.array(self.coefs)
        return self

    def _memberships(self, X):
        d2 = ((X[:, None, :] - self.centers[None]) ** 2).sum(-1)
        W = np.exp(-d2 / (2 * self.sigmas[None] ** 2 + 1e-9))
        return W / (W.sum(1, keepdims=True) + 1e-12)

    def predict(self, X):
        W = self._memberships(X)
        Xa = np.hstack([X, np.ones((len(X), 1))])
        rule_out = Xa @ self.coefs.T                 # [N, R]
        return (W * rule_out).sum(1)


def anfis_portfolio(prices: pd.DataFrame, t: int, lookback: int = 126,
                    horizon: int = 63, top_n: int = 10, cap: float = 0.30,
                    n_rules: int = 4, seed: int = 0) -> pd.Series:
    """ANFIS (Takagi-Sugeno) como comparador declarado en el anteproyecto."""
    from motor_owa.criteria import compute_criteria
    X, y = _features_and_target(prices, t, lookback, horizon)
    crit = compute_criteria(prices, t, lookback)
    if len(y) < 30:
        return equal_weight(prices.iloc[t - lookback:t].pct_change().dropna())
    mdl = _AnfisLite(n_rules=n_rules, seed=seed).fit(X, y)
    pred = pd.Series(mdl.predict(crit.values), index=crit.index)
    sel = pred.nlargest(top_n).index
    rets = prices[sel].iloc[t - lookback:t].pct_change().dropna()
    iv = 1.0 / np.maximum(rets.std().values, 1e-9)
    return pd.Series(_cap_norm(iv, cap), index=sel)
