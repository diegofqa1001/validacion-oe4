# validacion-oe4 — Validación interna del motor adaptativo (Objetivo 4)

Repositorio del **Objetivo Específico 4** de la tesis doctoral (UNAL Manizales):
*Estimar y validar internamente el modelo propuesto mediante simulaciones con
datos reales, utilizando métricas de precisión como el RMSE y estrategias de
partición de datos, con el fin de analizar su estabilidad predictiva.*
Respalda el Capítulo 7 de la monografía.

Es **independiente** del repositorio del motor
([motor-owa-v2](https://github.com/diegofqa1001/motor-owa-v2)), que usa como
dependencia: este repositorio contiene el *experimento*; aquel, el *artefacto*.

## Datos

Instantánea versionada en `data/snapshot_oe4/` (Yahoo Finance, endpoint
v8/finance/chart, cierre ajustado; 2015-01-01 a 2026-07-02; descarga del
26-sep-2026; generador `scripts/build_snapshot_oe4.py`; `MANIFEST.json` con
fecha de descarga, reglas y sumas SHA-256).

- **Colombia (BVC)**, mercado declarado en el anteproyecto: **17 emisores**
  (ECOPETROL, ISA, GEB, CIBEST, PFCIBEST, GRUPOSURA, GRUPOARGOS, CEMARGOS,
  NUTRESA, EXITO, PROMIGAS, CELSIA, BOGOTA, CORFICOLCF, PFDAVVNDA, MINEROS,
  TERPEL), 2 656 jornadas. Bancolombia se incluye mediante los símbolos
  vigentes de Grupo Cibest (`CIBEST.CL` y `PFCIBEST.CL`), bajo los cuales la
  fuente publica la historia completa desde 2015
  (`scripts/diag_simbolos_bancolombia.py`,
  `results/diag_simbolos_bancolombia.csv`).
- **Estados Unidos**, mercado de contraste y réplica: 25 grandes
  capitalizaciones del S&P 500, 2 891 jornadas.
- Regla de inclusión: al menos 80 % de observaciones válidas por activo.
- Calendario (regla de `repo_OWA/src/data.py`, Cap. 5): se descartan los días
  en que menos de la mitad de los emisores registra volumen positivo y los días
  en que todos repiten el precio del día anterior. En Colombia se retiran
  226 + 106 = 332 fechas (la fuente rellena los festivos de la BVC con el
  precio previo); en Estados Unidos, ninguna.

Ambos universos se componen de emisores vigentes a la fecha de descarga
(sesgo de supervivencia declarado en la tesis).

## Qué implementa (declarado en el anteproyecto)

1. **Protocolo**: ventanas rodantes causales de 126 días, horizonte de 63 días,
   costo de 10 pb por rebalanceo, partición cronológica **70-20-10**
   (40 ventanas en Colombia y 43 en Estados Unidos; 12 y 13 ventanas de
   verificación y validación para los comparadores).
2. **Métricas**: RMSE, MAE, MAPE, NDCG@k, MRR, consistencia ordinal (vía
   motor-owa-v2) y coherencia conductual Spearman(orness, σ).
3. **Comparadores**: 1/N, mínima varianza y máximo Sharpe (media-varianza,
   tope 30 %), **red neuronal** (MLP) y **ANFIS** (Takagi-Sugeno de primer
   orden), entrenados sin fuga de información; contrastes de Diebold-Mariano
   (HLN), corrección de Holm (`scripts/dm_holm.py`), no inferioridad unilateral
   y TOST con δ = 1 y 2 puntos porcentuales por trimestre.
4. **Estabilidad**: ruido gaussiano sobre los criterios, estrés (ventana de
   252 jornadas de máxima caída del promedio del universo) y sensibilidad al
   cambio de perfil (matriz de migraciones).
5. **Componente emocional (OE4-E)**: estudio de recuperación de parámetros con
   decisores simulados sobre el mercado estadounidense
   (`scripts/run_emocional.py`).

## Resultados principales (`results/`)

| Resultado | Colombia | Estados Unidos | Archivo |
|---|---|---|---|
| Spearman(orness, volatilidad realizada) | +1,000 | +1,000 | `{co,us}_coherencia_motor.csv` |
| Spearman(orness, retorno realizado) | −0,167 | +0,976 | `{co,us}_coherencia_motor.csv` |
| Coherencia en estrés | +0,310 (6-feb-2019 a 19-mar-2020) | +1,000 (4-ene-2022 a 5-ene-2023) | `{co,us}_estres.csv`, `{co,us}_estres_tramos.csv` |
| RMSE entre particiones | 0,125–0,173 | 0,109–0,132 | `{co,us}_metricas_motor.csv` |
| MAPE entre particiones | 14,9–69,4 % | 10,7–30,4 % | `{co,us}_metricas_motor.csv`, `diag_mape_denominador.csv` |
| NDCG@10 | 0,72–0,84 | 0,87–0,97 | `{co,us}_metricas_motor.csv` |
| Consistencia ordinal | 0,47–0,66 | 0,63–0,79 | `{co,us}_metricas_motor.csv` |
| Pares conservados con ruido 0,10 | 85,0 % | 89,5 % | `{co,us}_estabilidad_ruido.csv` |
| Contrastes DM significativos al 5 % | 2 de 8 (1 tras Holm sobre 16) | 0 de 8 | `{co,us}_diebold_mariano.csv`, `dm_holm.csv` |
| No inferioridad (δ = 1 p. p.) | 2 de 8 | 2 de 8 | `{co,us}_no_inferioridad.csv` |

En Colombia, los dos contrastes significativos favorecen al motor (Guardián
frente a mínima varianza, p = 0,002; Pragmático frente al MLP, p = 0,009).
El MAPE elevado del Guardián colombiano (69,4 %) procede de la escala baja de
sus puntajes (mediana 0,33) y se mantiene en 63,3 % al excluir el activo de
menor puntaje (`diag_mape_denominador.csv`). La reducción de la caída máxima
del Guardián frente a cada comparador está en `reduccion_riesgo_maximo.csv`.

## Uso

```bash
pip install -r requirements.txt
pip install -e ../motor-owa-v2          # dependencia
pytest                                   # pruebas
bash scripts/reproducir_todo.sh          # protocolo completo sobre data/snapshot_oe4
python scripts/run_oe4.py --market both  # resultados citables (CSV en results/)
python scripts/run_reduccion_riesgo.py   # reducción del riesgo máximo
python scripts/dm_holm.py                # corrección de Holm de los contrastes DM
python scripts/diag_mape_denominador.py  # diagnóstico del MAPE
python scripts/run_emocional.py          # experimento del componente emocional
python scripts/fig_comparacion_modelos.py  # comparadores (F29)
python scripts/fig_estres.py               # Figura 7.1: coherencia bajo estrés (F30)
python scripts/fig_migracion.py            # Figura 7.2: matriz de migración (F31)
python scripts/fig_emocional.py            # Figura 7.3: experimento OE4-E (F32)
```

## Cita

Quintero-Avellaneda, D. F. (2026). *validacion-oe4: validación interna del
motor adaptativo de recomendación* [Software y conjunto de datos]. GitHub.
https://github.com/diegofqa1001/validacion-oe4

## Licencia
MIT (código); CC-BY-4.0 (contenido).
