# validacion-oe4 — Validación interna del motor adaptativo (Objetivo 4)

Repositorio del **Objetivo Específico 4** de la tesis doctoral (UNAL Manizales):
*Estimar y validar internamente el modelo propuesto mediante simulaciones con
datos reales, utilizando métricas de precisión como el RMSE y estrategias de
partición de datos, con el fin de analizar su estabilidad predictiva.*

Es **independiente** del repositorio del motor
([motor-owa-v2](https://github.com/diegofqa1001/motor-owa-v2)), que usa como
dependencia: este repo contiene el *experimento*, aquel contiene el *artefacto*.

## Qué implementa (declarado en el anteproyecto)

1. **Datos reales** CO (BVC, 15 emisores) y US (25 blue chips), 2015–presente,
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

> **Calendario de negociación (2026-09-26).** El snapshot aplica la regla de
> `repo_OWA/src/data.py` (Cap. 5): se descartan (i) los días en que menos de la
> mitad de los emisores registra volumen positivo y (ii) los días en que todos
> los emisores repiten el precio del día anterior. En Colombia se descartan
> 216 + 106 fechas (la fuente rellena los festivos de la BVC con el precio
> previo); en EE. UU., ninguna. El panel CO pasa de 2 978 a 2 656 jornadas
> (45 → 40 ventanas; 14 → 12 en los comparadores). Volúmenes versionados en
> `data/snapshot_oe4/{us,co}_volumen.csv`; regla en `MANIFEST.json`. Las cifras
> CO de la nota siguiente quedan sustituidas por las de `results/`.

> **Reproducibilidad con datos versionados (2026-09-25).** Los precios de
> entrada quedan fijados en `data/snapshot_oe4/` (Yahoo Finance, endpoint
> v8/chart, cierre ajustado; 2015-01-01 a 2026-07-02; 25 activos US y 15 CO,
> con BCOLOMBIA.CL y PFBCOLOM.CL sin serie; `MANIFEST.json` con fecha de
> descarga, regla de inclusión y SHA-256; generador
> `scripts/build_snapshot_oe4.py`). `scripts/reproducir_todo.sh` re-ejecuta
> todo el protocolo y `scripts/comparar_con_archivo.py` contrasta el resultado
> con los CSV de julio de 2026, conservados en `results/archivo_2026-07/`
> (`results/comparacion_archivo_2026-07.csv`). Con el calendario del 26 de
> septiembre (nota anterior), reproducen dentro de la tolerancia de redondeo
> (5e-4) los registros, las métricas y la coherencia (+1,000) del motor en
> EE. UU., la matriz de migraciones (`sensibilidad_perfil.csv`) y el resumen
> del experimento OE4-E (`emocional_resumen.csv`, λ̂ = 2,17); en el detalle
> individual de OE4-E coinciden brechas, migraciones y riqueza, y difiere la
> λ̂ de algunos decisores. Todas las cifras de Colombia difieren, porque el
> panel pasa de 45 a 40 ventanas (p. ej., coherencia del motor: +0,929 en el
> archivo y +1,000 en la re-ejecución); las cifras citadas en la tesis son
> siempre las de la re-ejecución.
> Difieren y se sustituyen: (i) los comparadores (1/N, mínima varianza, máximo
> Sharpe, MLP y ANFIS) y los contrastes de Diebold-Mariano, porque el archivo
> de julio promedia 5 ventanas y la re-ejecución usa las 13 (EE. UU.) y 12
> (Colombia) ventanas de verificación y validación de la Tabla 7.3; la corrida
> ANFIS de julio, además, no es reproducible; (ii) la estabilidad ante ruido,
> cuyo archivo de julio no corresponde al código publicado; (iii) el estrés,
> porque la rejilla anterior omitía el último trimestre de la ventana
> (`stability.stress_grid`) y, en Colombia, porque el nuevo calendario cambia
> la ventana de peor caída del mercado: hoy es del 7-feb-2022 al 2-mar-2023,
> con coherencia en estrés de +0,429 (`results/co_estres.csv` y
> `results/co_estres_tramos.csv`), frente a +0,905 en el archivo;
> en EE. UU. la coherencia en estrés se mantiene en +1,000 y las volatilidades
> por perfil difieren a lo sumo en 0,008; (iv) `reduccion_riesgo_maximo.csv`, antes
> sin generador y con n = 5, ahora lo produce `scripts/run_reduccion_riesgo.py`
> sobre las 13 y 12 ventanas de la Tabla 7.3 (la reducción de caída máxima
> del Guardián frente a 1/N en EE. UU. pasa del 10,6 % del archivo a −2,3 %). Se añadió
> `{us,co}_no_inferioridad.csv` (no inferioridad unilateral y TOST con margen
> δ = 1 y 2 puntos porcentuales por trimestre). Las notas del 17 de agosto que
> siguen describen el estado de esa fecha: sus cifras (p. ej., el 10,6 %) y la
> mención de un `data/us_precios.csv` no versionado quedan sustituidas por
> `results/` y `data/snapshot_oe4/`.

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
bash scripts/reproducir_todo.sh          # protocolo completo sobre data/snapshot_oe4
python scripts/run_oe4.py --market both  # resultados citables (CSV en results/)
python scripts/run_emocional.py          # experimento del componente emocional
python scripts/fig_comparacion_modelos.py  # Figura §7.4: comparadores (F29)
python scripts/fig_estres.py               # Figura 7.1: coherencia bajo estrés (F30)
python scripts/fig_migracion.py            # Figura 7.2: matriz de migración (F31)
python scripts/fig_emocional.py            # Figura 7.3: experimento OE4-E (F32)
```

## Licencia
MIT (código); CC-BY-4.0 (contenido).
