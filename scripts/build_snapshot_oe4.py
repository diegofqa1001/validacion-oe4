"""Construye el snapshot versionado de precios del OE4 (data/snapshot_oe4/).

Motivo: los resultados de results/ se produjeron con descargas en vivo de
Yahoo Finance (motor_owa.data.load_yfinance, end=None) que no quedaron
versionadas. Este script fija los datos de entrada:

  1. Descarga, para cada ticker de los universos TICKERS_US y TICKERS_CO de
     motor-owa-v2 (config.py), la serie diaria del endpoint publico
     https://query2.finance.yahoo.com/v8/finance/chart/<ticker>
     (cierre ajustado por dividendos y splits = 'adjclose', equivalente a
     yfinance.download(auto_adjust=True)['Close']).
  2. Guarda la respuesta cruda por ticker (raw/<ticker>.csv: fecha local de
     la bolsa, close, adjclose, volume).
  3. Ensambla el panel con la MISMA regla de inclusion de load_yfinance:
     union de fechas en [START, END], se descartan los activos con menos
     del 80 % de observaciones validas, ffill y dropna de filas iniciales.
  4. Escribe {us,co}_precios.csv y MANIFEST.json con la fecha de descarga,
     el periodo, los activos incluidos/excluidos y el SHA-256 de cada archivo.

Uso:
  python scripts/build_snapshot_oe4.py            # descarga + ensamblado
  python scripts/build_snapshot_oe4.py --offline  # solo re-ensambla desde raw/
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "motor-owa-v2", "src"))
from motor_owa.config import TICKERS_CO, TICKERS_US  # noqa: E402

OUT = os.path.join(HERE, "..", "data", "snapshot_oe4")
RAW = os.path.join(OUT, "raw")
START = "2015-01-01"
# Fin del periodo: ultima jornada de la corrida archivada. Se fija de modo que
# el panel US tenga las 2 891 jornadas del Cap. 7 (§7.1) y verifique los
# anclajes de fecha de los CSV archivados (us_estres_trayectoria.csv:
# indice 1764 = 2022-01-04; co_estres_trayectoria.csv: indice 1108 =
# 2019-04-02). Ver MANIFEST.json -> anclajes.
END = "2026-07-02"
URL = ("https://query2.finance.yahoo.com/v8/finance/chart/{t}?period1={p1}"
       "&period2={p2}&interval=1d&events=div%2Csplit&includeAdjustedClose=true")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(ticker: str) -> pd.DataFrame:
    p1 = int(pd.Timestamp("2014-12-01", tz="UTC").timestamp())
    p2 = int(time.time())
    req = urllib.request.Request(URL.format(t=ticker, p1=p1, p2=p2),
                                 headers={"User-Agent": UA,
                                          "Accept": "application/json"})
    for intento in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                js = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:  # ticker sin serie en Yahoo (deslistado)
                print(f"  {ticker}: 404 (sin serie disponible)")
                return None
            espera = 5 * (intento + 1)
            print(f"  {ticker}: {e}; reintento en {espera}s")
            time.sleep(espera)
        except Exception as e:  # transitorios
            espera = 5 * (intento + 1)
            print(f"  {ticker}: {e}; reintento en {espera}s")
            time.sleep(espera)
    else:
        raise RuntimeError(f"no se pudo descargar {ticker}")
    res = js["chart"]["result"][0]
    tz = res["meta"]["exchangeTimezoneName"]
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz)
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    df = pd.DataFrame({"Date": ts.tz_localize(None).normalize(),
                       "close": q["close"], "adjclose": adj,
                       "volume": q["volume"]})
    df = df.dropna(subset=["adjclose"]).drop_duplicates("Date", keep="last")
    return df


def assemble(tickers, start=START, end=END):
    series, vols = {}, {}
    for t in tickers:
        f = os.path.join(RAW, f"{t}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
        series[t] = d["adjclose"]
        vols[t] = d["volume"]
    raw = pd.DataFrame(series).sort_index()
    raw = raw.loc[(raw.index >= start) & (raw.index <= end)]
    raw = raw.dropna(how="all")
    # Fechas sin negociacion (mismo criterio que repo_OWA/src/data.py, Cap. 5):
    # se descartan los dias en que menos de la mitad de los emisores registra
    # volumen positivo. En la BVC el endpoint rellena los festivos con el
    # precio previo y volumen nulo, lo que inyectaria rendimientos nulos.
    vol = pd.DataFrame(vols).reindex(raw.index).fillna(0.0)
    activo = (vol > 0).mean(axis=1) >= 0.5
    n_sin_neg = int((~activo).sum())
    raw, vol = raw[activo], vol[activo]
    # misma regla que motor_owa.data.load_yfinance
    px = raw.dropna(axis=1, thresh=int(0.8 * len(raw))).ffill().dropna()
    # (ii) Dias sin informacion nueva: todos los emisores repiten el precio del
    # dia anterior (rendimiento nulo en todos; excluye la primera fila). Misma
    # regla y mismo orden que repo_OWA/src/data.py.
    r = px.pct_change()
    nulo = r.notna().all(axis=1) & (r.abs().sum(axis=1) == 0)
    n_repetidos = int(nulo.sum())
    px = px[~nulo]
    px.index.name = "Date"
    vol = vol.reindex(index=px.index, columns=px.columns)
    vol.index.name = "Date"
    excl = {t: (int(raw[t].notna().sum()) if t in raw.columns else 0)
            for t in tickers if t not in px.columns}
    return px, raw, excl, vol, n_sin_neg, n_repetidos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    fecha = None
    no_disp = []
    if not args.offline:
        fecha = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        for t in TICKERS_US + TICKERS_CO:
            print(f"[snapshot] {t}")
            df = fetch(t)
            if df is None:
                no_disp.append(t)
                continue
            df.to_csv(os.path.join(RAW, f"{t}.csv"), index=False)
        with open(os.path.join(RAW, "_descarga.json"), "w") as f:
            json.dump({"fecha_descarga_utc": fecha,
                       "sin_serie_en_yahoo": no_disp}, f)
            time.sleep(1.0)
    man_path = os.path.join(OUT, "MANIFEST.json")
    prev = json.load(open(os.path.join(RAW, "_descarga.json")))
    man = {"fuente": "Yahoo Finance, endpoint v8/finance/chart (adjclose)",
           "fecha_descarga_utc": fecha or prev.get("fecha_descarga_utc"),
           "periodo": {"inicio": START, "fin": END},
           "regla_fechas": ("se descartan (i) los dias en que menos de la mitad "
                            "de los emisores registra volumen positivo y (ii) "
                            "los dias en que todos los emisores repiten el "
                            "precio del dia anterior (rendimiento nulo en "
                            "todos; excluye la primera fila); regla identica a "
                            "repo_OWA/src/data.py (Cap. 5). La fuente rellena "
                            "los festivos de la BVC con el precio previo."),
           "regla_inclusion": ("union de fechas de negociacion; se excluye el "
                               "activo con < 80 % de observaciones validas en "
                               "el periodo; ffill; se descartan las filas "
                               "iniciales incompletas (motor_owa.data."
                               "load_yfinance)"),
           "sin_serie_en_yahoo": prev.get("sin_serie_en_yahoo", []),
           "mercados": {}, "archivos": {}}
    for mkt, tick in (("us", TICKERS_US), ("co", TICKERS_CO)):
        px, raw, excl, vol, n_sin_neg, n_repetidos = assemble(tick)
        path = os.path.join(OUT, f"{mkt}_precios.csv")
        px.to_csv(path)
        vol.to_csv(os.path.join(OUT, f"{mkt}_volumen.csv"))
        man["mercados"][mkt] = {
            "n_jornadas": int(len(px)), "n_activos": int(px.shape[1]),
            "primera_fecha": str(px.index[0].date()),
            "ultima_fecha": str(px.index[-1].date()),
            "activos": list(px.columns),
            "excluidos_obs_validas": excl,
            "fechas_sin_negociacion_descartadas": n_sin_neg,
            "fechas_precio_repetido_descartadas": n_repetidos}
    for root, _, files in os.walk(OUT):
        for fn in sorted(files):
            if fn == "MANIFEST.json":
                continue
            p = os.path.join(root, fn)
            man["archivos"][os.path.relpath(p, OUT)] = sha256(p)
    json.dump(man, open(man_path, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(man["mercados"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
