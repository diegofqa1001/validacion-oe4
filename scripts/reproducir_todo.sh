#!/bin/bash
# Re-ejecuta el protocolo OE4 completo sobre data/snapshot_oe4 (orden fijo).
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OE4_BACKEND=${OE4_BACKEND:-numpy}
cd "$(dirname "$0")/.."
python3 scripts/run_oe4.py --market both
for m in us co; do
  python3 scripts/run_comparadores_ampliado.py $m ve
  python3 scripts/run_comparadores_ampliado.py $m va
  python3 scripts/run_comparadores_ampliado.py $m resumen
done
python3 scripts/run_reduccion_riesgo.py
python3 scripts/dm_holm.py
python3 scripts/run_emocional.py
