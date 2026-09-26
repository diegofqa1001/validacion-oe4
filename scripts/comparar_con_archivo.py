"""Compara los resultados re-ejecutados sobre data/snapshot_oe4 con los CSV
archivados de julio de 2026 (results/archivo_2026-07/).

Para cada archivo comun, informa la maxima diferencia absoluta de las
columnas numericas (alineando por indice/filas) y si cae dentro de la
tolerancia de redondeo de las cifras publicadas (5e-4: medio punto en el
tercer decimal, o 0,05 puntos porcentuales).

Salida: results/comparacion_archivo_2026-07.csv
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ARC = os.path.join(RES, "archivo_2026-07")
TOL = 5e-4
KEYS = {"comparadores": "modelo", "comparadores_ampliado": "modelo",
        "metricas_motor": 0, "sensibilidad_perfil": "perfil",
        "emocional_resumen": "grupo", "estres_trayectoria": "date"}
rows = []
for fn in sorted(os.listdir(ARC)):
    new = os.path.join(RES, fn)
    old = os.path.join(ARC, fn)
    if not fn.endswith(".csv"):
        continue
    if not os.path.exists(new):
        rows.append({"archivo": fn, "estado": "sin equivalente en la re-ejecucion"})
        continue
    a, b = pd.read_csv(old), pd.read_csv(new)
    key = next((v for k, v in KEYS.items() if fn.endswith(k + ".csv")), None)
    if key is not None:
        col = a.columns[key] if isinstance(key, int) else key
        a, b = a.set_index(col), b.set_index(col)
        idx = a.index.intersection(b.index)
        a, b = a.loc[idx], b.loc[idx]
    elif {"t", "modelo"} <= set(a.columns):
        a, b = a.set_index(["t", "modelo"]), b.set_index(["t", "modelo"])
        idx = a.index.intersection(b.index)
        a, b = a.loc[idx], b.loc[idx]
    elif {"slice", "t", "profile"} <= set(a.columns):
        a, b = a.set_index(["slice", "t", "profile"]), b.set_index(["slice", "t", "profile"])
        idx = a.index.intersection(b.index)
        a, b = a.loc[idx], b.loc[idx]
    cols = [c for c in a.columns if c in b.columns
            and pd.api.types.is_numeric_dtype(a[c])
            and pd.api.types.is_numeric_dtype(b[c])]
    n = min(len(a), len(b))
    if n == 0 or not cols:
        rows.append({"archivo": fn, "estado": "sin filas o columnas comunes"})
        continue
    d = (a[cols].iloc[:n].reset_index(drop=True)
         - b[cols].iloc[:n].reset_index(drop=True)).abs()
    worst = d.max().sort_values(ascending=False)
    rows.append({"archivo": fn, "filas_archivo": len(a), "filas_nuevas": len(b),
                 "max_dif_abs": float(worst.iloc[0]),
                 "columna_max_dif": worst.index[0],
                 "estado": ("reproduce" if worst.iloc[0] <= TOL
                            else "difiere")})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "comparacion_archivo_2026-07.csv"), index=False)
print(out.to_string(index=False))
