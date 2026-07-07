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

__all__ = ["diebold_mariano"]


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
