"""Diagnostico de los simbolos de Bancolombia en la fuente de datos.

Bancolombia forma parte de los universos colombianos. Tras su reorganizacion
societaria como Grupo Cibest, la fuente (Yahoo Finance) publica su historia
completa desde 2015 bajo los simbolos vigentes CIBEST.CL (ordinaria) y
PFCIBEST.CL (preferencial), mientras que los simbolos historicos BCOLOMBIA.CL
y PFBCOLOM.CL ya no devuelven serie. Por eso validacion-oe4 (Cap. 7) incluye
a Bancolombia mediante CIBEST.CL y PFCIBEST.CL
(data/snapshot_oe4/MANIFEST.json) y repo_OWA (Cap. 5 y apartado 8.4) mediante
PFCIBEST.CL. Este guion consulta el mismo endpoint (Yahoo Finance
v8/finance/chart, frecuencia mensual) para los simbolos historicos y los
vigentes (incluido el ADR CIB) y registra si hay serie, su primera y ultima
fecha y el nombre que la fuente asigna. Documenta la eleccion de simbolos;
no modifica ningun resultado.

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
