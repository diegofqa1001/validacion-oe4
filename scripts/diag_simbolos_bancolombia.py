"""Diagnostico de la ausencia de Bancolombia en los universos colombianos.

Los universos colombianos solicitaron simbolos historicos de Bancolombia:
repo_OWA (Cap. 5 y §8.4) solicito PFBCOLOM.CL (preferencial;
repo_OWA/data/snapshot_2026-08-31/MANIFEST_precios_origen.md) y
validacion-oe4 (Cap. 7) solicito BCOLOMBIA.CL (ordinaria) y PFBCOLOM.CL
(data/snapshot_oe4/MANIFEST.json, "sin_serie_en_yahoo"). La fuente no
devolvio serie para ninguno de ellos. Este guion
consulta el mismo endpoint (Yahoo Finance v8/finance/chart, frecuencia
mensual) para los simbolos historicos y para los simbolos vigentes del emisor
tras su reorganizacion como Grupo Cibest (CIBEST.CL, PFCIBEST.CL y el ADR
CIB), y registra si hay serie, su primera y ultima fecha y el nombre que la
fuente asigna. No modifica ningun resultado de la tesis.

Uso: python scripts/diag_simbolos_bancolombia.py
Salida: results/diag_simbolos_bancolombia.csv
"""
import datetime as dt
import os

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "diag_simbolos_bancolombia.csv")
UA = "Mozilla/5.0"
P1 = int(pd.Timestamp("2015-01-01", tz="UTC").timestamp())
P2 = int(pd.Timestamp("2026-07-02", tz="UTC").timestamp())
SIMBOLOS = ["BCOLOMBIA.CL", "PFBCOLOM.CL", "CIBEST.CL", "PFCIBEST.CL", "CIB"]

rows = []
consulta = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
for s in SIMBOLOS:
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{s}"
    r = requests.get(url, params={"period1": P1, "period2": P2,
                                  "interval": "1mo"},
                     headers={"User-Agent": UA}, timeout=30)
    ch = r.json()["chart"]
    res = (ch.get("result") or [None])[0]
    if not res or not res.get("timestamp"):
        err = (ch.get("error") or {}).get("description", "sin resultado")
        rows.append({"simbolo": s, "con_serie": False, "n_meses": 0,
                     "primera_fecha": "", "ultima_fecha": "", "nombre": "",
                     "bolsa": "", "mensaje_fuente": err,
                     "consulta_utc": consulta})
        continue
    ts = pd.to_datetime(res["timestamp"], unit="s")
    rows.append({"simbolo": s, "con_serie": True, "n_meses": len(ts),
                 "primera_fecha": ts.min().date().isoformat(),
                 "ultima_fecha": ts.max().date().isoformat(),
                 "nombre": res["meta"].get("longName", ""),
                 "bolsa": res["meta"].get("exchangeName", ""),
                 "mensaje_fuente": "", "consulta_utc": consulta})

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(df.to_string())
