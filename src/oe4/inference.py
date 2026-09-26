"""Inferencia formal para la comparacion de modelos (OE4).

Test de Diebold-Mariano (1995) sobre diferenciales de retorno por ventana,
con varianza de largo plazo Newey-West y la correccion de muestra pequena
de Harvey, Leybourne y Newbold (1997), adecuada para el numero moderado
de ventanas fuera de muestra del protocolo.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

__all__ = ["diebold_mariano", "noninferiority_dm"]


def _newey_west_lrv(d: np.ndarray, lag: int) -> float:
    n = len(d)
    d = d - d.mean()
    g0 = float(np.dot(d, d)) / n
    s = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(np.dot(d[k:], d[:-k])) / n
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    return max(s, 1e-12)


def _t_cdf(t: float, df: int) -> float:
    """CDF t-Student via aproximacion por funcion beta incompleta (sin scipy)."""
    # relacion con la beta incompleta regularizada
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    # betainc por integracion numerica de Gauss-Legendre simple
    xs, ws = np.polynomial.legendre.leggauss(64)
    lo, hi = 0.0, x
    u = 0.5 * (xs + 1) * (hi - lo) + lo
    val = np.sum(ws * (u ** (a - 1)) * ((1 - u) ** (b - 1))) * 0.5 * (hi - lo)
    beta_ab = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
    ibeta = float(val / beta_ab)
    p_two = ibeta          # P(|T|>|t|) aproximado
    cdf = 1.0 - 0.5 * p_two if t >= 0 else 0.5 * p_two
    return min(max(cdf, 0.0), 1.0)


def diebold_mariano(ret_a: Sequence[float], ret_b: Sequence[float],
                    lag: int | None = None) -> Tuple[float, float]:
    """DM sobre el diferencial de retornos por ventana (A - B).

    Devuelve (estadistico DM corregido HLN, p-valor bilateral).
    DM > 0: A supera a B en retorno medio por ventana.
    """
    a = np.asarray(ret_a, float).ravel()
    b = np.asarray(ret_b, float).ravel()
    if a.size != b.size or a.size < 4:
        raise ValueError("Se requieren >= 4 ventanas emparejadas.")
    d = a - b
    n = d.size
    if lag is None:
        lag = max(1, int(round(4 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = _newey_west_lrv(d, lag)
    dm = d.mean() / math.sqrt(lrv / n)
    # correccion Harvey-Leybourne-Newbold para muestras pequenas
    h = 1  # horizonte de pronostico en unidades de ventana
    c = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * c
    p = 2.0 * (1.0 - _t_cdf(abs(dm_hln), n - 1))
    return float(dm_hln), float(min(max(p, 0.0), 1.0))


def _t_ppf(q: float, df: int) -> float:
    """Cuantil t-Student por biseccion sobre _t_cdf (sin scipy)."""
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def noninferiority_dm(ret_a: Sequence[float], ret_b: Sequence[float],
                      delta: float, alpha: float = 0.05,
                      lag: int | None = None) -> dict:
    """No inferioridad y equivalencia (TOST) de A frente a B con margen delta.

    Mismo estadistico que diebold_mariano (diferencial d = A - B por ventana,
    varianza de largo plazo Newey-West y correccion HLN), aplicado a las
    hipotesis desplazadas por el margen:

    * No inferioridad: H0: E[d] <= -delta  vs  H1: E[d] > -delta
      (contraste unilateral; p_ni < alpha => A no es inferior a B en mas
      de delta por ventana).
    * Equivalencia (TOST, Schuirmann, 1987): ademas H0': E[d] >= +delta;
      p_tost = max(p_ni, p_sup).

    Devuelve tambien el limite inferior unilateral (1 - alpha) de E[d] y el
    margen minimo delta* = max(0, -limite) con el que se declararia no
    inferioridad a ese nivel.
    """
    a = np.asarray(ret_a, float).ravel()
    b = np.asarray(ret_b, float).ravel()
    d = a - b
    n = d.size
    if n < 4:
        raise ValueError("Se requieren >= 4 ventanas emparejadas.")
    if lag is None:
        lag = max(1, int(round(4 * (n / 100.0) ** (2.0 / 9.0))))
    se = math.sqrt(_newey_west_lrv(d, lag) / n)
    h = 1
    c = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    se_eff = se / c                       # error tipico con correccion HLN
    t_ni = (d.mean() + delta) / se_eff
    t_sup = (d.mean() - delta) / se_eff
    p_ni = 1.0 - _t_cdf(t_ni, n - 1)
    p_sup = _t_cdf(t_sup, n - 1)
    q = _t_ppf(1.0 - alpha, n - 1)
    lb = d.mean() - q * se_eff
    ub = d.mean() + q * se_eff
    return {"media_dif": float(d.mean()), "ee_hln": float(se_eff),
            "t_no_inf": float(t_ni), "p_no_inf": float(p_ni),
            "p_tost": float(max(p_ni, p_sup)),
            "lim_inf_unilateral": float(lb), "lim_sup_unilateral": float(ub),
            "delta_min_no_inf": float(max(0.0, -lb)), "n": int(n)}
