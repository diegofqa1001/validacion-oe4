"""Ejecuta el protocolo OE4 con datos reales CO y US.

Uso: python scripts/run_oe4.py [--market co|us|both] [--quick]
Requiere el paquete motor-owa-v2 (carpeta hermana o instalado con pip).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
# motor-owa-v2 como carpeta hermana (o instalado via pip)
sys.path.insert(0, os.path.join(HERE, "..", "..", "motor-owa-v2", "src"))

from oe4.pipeline import run_market


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["co", "us", "both"], default="both")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--snapshot", default=os.path.join(HERE, "..", "data",
                                                       "snapshot_oe4"),
                    help="carpeta con {co,us}_precios.csv versionados; "
                         "'yahoo' fuerza la descarga en vivo (no citable)")
    ap.add_argument("--outdir", default=os.path.join(HERE, "..", "results"))
    args = ap.parse_args()

    from motor_owa.data import load_yfinance, load_csv, TICKERS_CO, TICKERS_US
    outdir = args.outdir
    todo = {"co": TICKERS_CO, "us": TICKERS_US}
    if args.market != "both":
        todo = {args.market: todo[args.market]}
    for mkt, tickers in todo.items():
        if args.snapshot == "yahoo":
            print(f"[oe4] descargando {mkt.upper()} ({len(tickers)} activos)...")
            px = load_yfinance(tickers, start=args.start)
        else:
            px = load_csv(os.path.join(args.snapshot, f"{mkt}_precios.csv"))
        print(f"[oe4] {mkt.upper()}: {px.shape[0]} dias x {px.shape[1]} activos")
        res = run_market(px, mkt, outdir, quick=args.quick)
        print(res["comparadores"].round(4))
        print(f"[oe4] coherencia estres: "
              f"{res['estres']['stress_coherence_vol']:+.3f}")


if __name__ == "__main__":
    main()
