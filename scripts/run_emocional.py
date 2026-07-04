"""Experimento OE4-E: validacion del componente emocional del decisor.

Pregunta: ¿puede el motor DETECTAR que la reclasificacion del inversor no
es puramente logica, y ESTIMAR sus parametros conductuales a partir de las
re-respuestas al cuestionario?

Diseno (validacion interna, sin humanos): dos poblaciones de decisores
sinteticos atraviesan el mercado real US con re-elicitacion declarada en
cada horizonte:
  - LOGICOS   (emotion_gain = 0): reaccionan solo a la sorpresa (control).
  - EMOCIONALES (emotion_gain = 0.5): ademas reaccionan a su sentimiento
    (euforia/panico alineado con el signo de la sorpresa reciente).
Se reporta: brecha emocional media |eps|, correlacion brecha-sorpresa y
lambda estimado, por poblacion, CENSURANDO los ciclos saturados del
instrumento (|z| >= 1.45: la escala Likert acotada no puede declarar mas
alla del octil extremo; censura estandar). Exito del mecanismo: separar poblaciones
y recuperar lambda ~ 2.25 en los logicos.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "motor-owa-v2", "src"))

import numpy as np
import pandas as pd
from motor_owa.adaptive import InvestorState
from motor_owa.config import EngineConfig
from motor_owa.engine import RecommendationEngine
from motor_owa.elicitation import simulate_declared_scores, emotional_gap_metrics


def run(prices, n_by_group=30, seed=20260704, profile0="Pragmatist"):
    cfg = EngineConfig()
    eng = RecommendationEngine(prices, cfg)
    rng = np.random.default_rng(seed)
    rows = []
    t0 = cfg.lookback
    n_cycles = min(12, (len(prices) - t0) // cfg.horizon - 1)
    for group, gain in [("logico", 0.0), ("emocional", 0.5)]:
        for i in range(n_by_group):
            st = InvestorState.from_profile(profile0)
            t = t0 + int(rng.integers(0, cfg.horizon))  # desfase de entrada
            sent = 0.0
            for _ in range(n_cycles):
                if t + cfg.horizon >= len(prices):
                    break
                port = eng.builder.build(st.profile, t)
                r = eng.realized_return(port.weights, t, cfg.horizon)
                mu_h = port.expected_return * cfg.horizon / 252
                sig_h = port.expected_vol * np.sqrt(cfg.horizon / 252)
                s = (r - mu_h) / max(sig_h, 1e-9)
                sent = 0.6 * sent + 0.4 * np.sign(s)   # sentimiento con memoria
                dec = simulate_declared_scores(st.z, s, cfg.kappa,
                                               cfg.loss_lambda,
                                               emotion_gain=gain,
                                               sentiment=sent,
                                               noise_sd=0.15, rng=rng)
                from motor_owa.adaptive import harvest_and_recalibrate
                harvest_and_recalibrate(st, r, mu_h, sig_h, cfg,
                                        declared_scores=dec)
                t += cfg.horizon
            m = emotional_gap_metrics(st.history, z_bound=1.45)
            migr = sum(1 for h in st.history if h["migrated"])
            rows.append({"grupo": group, "inversor": i, **m,
                         "migraciones": migr,
                         "perfil_final": st.profile.name,
                         "riqueza_final": st.wealth})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    px = pd.read_csv(os.path.join(HERE, "..", "data", "us_precios.csv"),
                     parse_dates=["Date"], index_col="Date")
    df = run(px)
    out = os.path.join(HERE, "..", "results")
    df.to_csv(os.path.join(out, "emocional_individual.csv"), index=False)
    resumen = df.groupby("grupo").agg(
        n=("inversor", "size"),
        brecha_abs_media=("mean_abs_gap", "mean"),
        corr_brecha_sorpresa=("corr_gap_surprise", "mean"),
        lambda_estimado=("lambda_hat", "median"),
        migraciones_medias=("migraciones", "mean"))
    resumen.to_csv(os.path.join(out, "emocional_resumen.csv"))
    print(resumen.round(3).to_string())
