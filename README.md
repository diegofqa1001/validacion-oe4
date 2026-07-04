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

## Uso

```bash
pip install -r requirements.txt
pip install -e ../motor-owa-v2          # dependencia
pytest                                   # 8 pruebas
python scripts/run_oe4.py --market both # resultados citables (CSV en results/)
python scripts/run_emocional.py          # experimento del componente emocional
```

## Licencia
MIT (código); CC-BY-4.0 (contenido).
