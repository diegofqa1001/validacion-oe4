"""Rotulado en espanol compartido por las figuras del Cap. 7."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "motor-owa-v2", "src"))
from motor_owa.viz_es import (NOMBRES_ES, COMPARADORES_ES, OKABE_ITO, coma,  # noqa
                              ejes_coma, estilo, guardar)
from matplotlib.colors import LinearSegmentedColormap

#: divergente Okabe-Ito: bermellon (hacia la prudencia) - blanco - azul
DIVERGENTE_OI = LinearSegmentedColormap.from_list(
    "oi_div", ["#D55E00", "#FFFFFF", "#0072B2"])
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep",
         "oct", "nov", "dic"]


def fecha_es(ts):
    return f"{ts.day} {MESES[ts.month - 1]} {ts.year}"
