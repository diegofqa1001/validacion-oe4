# validacion-oe4 — Validación interna del motor adaptativo (Objetivo 4)

Repositorio del **Objetivo Específico 4** de la tesis doctoral (UNAL Manizales):
*Estimar y validar internamente el modelo propuesto mediante simulaciones con
datos reales, utilizando métricas de precisión como el RMSE y estrategias de
partición de datos, con el fin de analizar su estabilidad predictiva.*

Es **independiente** del repositorio del motor
([motor-owa-v2](https://github.com/diegofqa1001/motor-owa-v2)), que usa como
dependencia: este repo contiene el *experimento*, aquel contiene el *artefacto*.

## Qué implementa (declarado en el anteproyecto)

1. **Datos reales** CO (BVC, 17 emisores) y US (25 blue chips), 2015–presente,
   ventanas rodantes sin look-ahead, partición **70-20-10**.
2. **Métricas**: RMSE, MAE, MAPE, NDCG@k, MRR, consistencia ordinal (vía
   motor-owa-v2) + coherencia conductual Spearman(orness, σ).
3. **Comparadores**: 1/N, mínima varianza y máximo Sharpe (media-varianza,
   SLSQP con tope 30 %), **red neuronal** (MLP 16×8) y **ANFIS**
   (Takagi-Sugeno de primer orden: reglas k-means + pertenencias gaussianas +
   consecuentes lineales), entrenados sin fuga de información.
4. **Estabilidad**: perturbaciones de ruido sobre los criterios (consistencia
   ordinal por nivel de ruido), estrés (peor subperiodo de caída del mercado)
   y sensibilidad al cambio de perfil (matriz de migraciones).
5. **Validación del componente emocional (OE4-E)**: dos poblaciones de
   decisores sintéticos (lógicos vs. emocionales) con re-elicitación
   declarada sobre el mercado real US; el pipeline separa las poblaciones
   por la brecha emocional ε y recupera la aversión a la pérdida sembrada
   (λ̂ ≈ 2.17 vs. 2.25) en el grupo de control (`scripts/run_emocional.py`).

> **Nota de verificación (añadida 2026-08-17).** Los cinco comparadores del
> §7.4 de la tesis (1/N, mínima varianza, máximo Sharpe, MLP, ANFIS) y el test
> de Diebold-Mariano (Tablas 7.3-7.4) ya estaban implementados y se
> reverificaron cifra por cifra contra `results/reduccion_riesgo_maximo.csv`
> y `results/{us,co}_diebold_mariano.csv`: coinciden exactamente con el texto
> (p. ej. reducción de caída máxima del Guardian frente a 1/N: 10,6 %; cero de
> los 16 contrastes de Diebold-Mariano es significativo al 5 %). Lo que
> **faltaba** era la evidencia reproducible de tres figuras que ya estaban
> incorporadas a la tesis (7.1, 7.2, 7.3): existían los datos pero no el
> script que los convierte en figura, y en el caso de la Figura 7.1 el propio
> `pipeline.py` calculaba la volatilidad por perfil bajo estrés
> (`stress_mean_vols`) y la descartaba sin persistirla. Se corrigió
> `src/oe4/pipeline.py` para guardar el detalle completo del subperiodo de
> estrés (`results/{market}_estres.csv`, antes solo `stress_coherence_vol`
> sobrevivía en `{market}_coherencia.csv`) y se añadieron
> `scripts/fig_estres.py`, `scripts/fig_migracion.py` y
> `scripts/fig_emocional.py`, que reproducen las Figuras 7.1-7.3 a partir de
> los CSV ya existentes (`{market}_estres.csv`, `sensibilidad_perfil.csv`,
> `emocional_individual.csv`/`emocional_resumen.csv`). Las imágenes
> resultantes se verificaron contra las incorporadas en la tesis: coinciden
> en cifras y en forma. Pendiente declarado (no bloqueante): `run_emocional.py`
> depende de un snapshot `data/us_precios.csv` que no está versionado en este
> repositorio; los CSV de resultados ya publicados en `results/` sí lo están
> y son la fuente citable mientras ese snapshot no se incorpore.

> **Actualización (2026-08-17, a pedido del autor).** La Figura 7.1 pasó de
> un diagrama de barras (volatilidad realizada *promedio* por perfil durante
> el estrés) a un diagrama de líneas: la trayectoria de valor acumulado
> día a día de las ocho carteras de perfil, superpuesta con la de los cinco
> comparadores del anteproyecto (1/N, mínima varianza, máximo Sharpe, MLP,
> ANFIS) sobre la MISMA ventana de estrés — así la comparación ya no es solo
> entre perfiles, sino también entre perfiles y modelos econométricos.
> `oe4.stability.stress_trajectories` encadena el retorno realizado día a día
> en la misma rejilla de rebalanceo que `stress_coherence` (evita doble
> cómputo y garantiza que ambas figuras describan el mismo subperiodo);
> `pipeline.py` persiste el resultado en
> `results/{market}_estres_trayectoria.csv` y `scripts/fig_estres.py` se
> reescribió para leer de ahí. La cifra de coherencia de Spearman en el
> título no cambió: describe el promedio de volatilidad del subperiodo, no
> la trayectoria diaria, y ambas fuentes son consistentes entre sí.

## Uso

```bash
pip install -r requirements.txt
pip install -e ../motor-owa-v2          # dependencia
pytest                                   # 8 pruebas
python scripts/run_oe4.py --market both  # resultados citables (CSV en results/)
python scripts/run_emocional.py          # experimento del componente emocional
python scripts/fig_comparacion_modelos.py  # Figura §7.4: comparadores (F29)
python scripts/fig_estres.py               # Figura 7.1: coherencia bajo estrés (F30)
python scripts/fig_migracion.py            # Figura 7.2: matriz de migración (F31)
python scripts/fig_emocional.py            # Figura 7.3: experimento OE4-E (F32)
```

## Licencia
MIT (código); CC-BY-4.0 (contenido).
