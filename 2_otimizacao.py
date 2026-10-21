# -*- coding: utf-8 -*-
"""
PASSO 2 -- OTIMIZACAO DOS GANHOS NO GEMEO (minimiza a funcao custo J)

Roda o firmware escolhido no gemeo calibrado sobre um roteiro de degraus
(200 -> 80 -> 250 -> 200 -> 120 -> 250, 15 s cada), calcula J (metricas.py) e
deixa a evolucao diferencial procurar Kp, Ki, Kd (e beta, no v8).

Depois testa a robustez: o melhor conjunto e' reavaliado em variacoes do
modelo (outra calibracao, atraso maior, mais ruido, K maior). Um ganho que so'
e' bom num modelo exato nao serve para a bancada.

Tempo: ~5 min.
"""
import os
import numpy as np
from scipy.optimize import differential_evolution
import modelo as M
import metricas as MT
import ensaios as E

# configuracao
PARAMETROS = os.path.join(E.PASTA_PARAM, "fis_rodada5.npy")   # gemeo usado
VARIANTE = 2                       # 1 = v7 (otimiza P,I,D) | 2 = v8 (P,I,D,beta)
LIMITES = [(0.03, 0.6), (0.0, 0.12), (0.0, 1.0), (0.5, 0.97)]  # Kp, Ki, Kd, beta
SEMENTES = range(5)                # realizacoes do ruido por avaliacao
ITERACOES = 40
CUSTO = "J2"                       # "J" (rodadas 1-4, usa ts) ou "J2" (usa IAE no lugar de ts)
COMPARAR = {"rodada 2 (0,42/0,08/0,36/0,95)": [0.42, 0.08, 0.36, 0.95],
            "rodada 3 (0,26/0,01/0,35/0,95)": [0.26, 0.01, 0.35, 0.95]}

FIS, S0 = M.carrega_fis(PARAMETROS)
ROTEIRO = [200, 80, 250, 200, 120, 250]
SP = np.repeat(np.array(ROTEIRO, float), int(15 / M.TS))


def metricas_gemeo(g, var=VARIANTE, fis=None, s0=None, sementes=SEMENTES):
    """Mediana das metricas por degrau sobre varias realizacoes do ruido."""
    fis = FIS if fis is None else fis
    s0 = S0 if s0 is None else s0
    beta = g[3] if len(g) > 3 else 0.0
    ctrl = np.array([g[0], g[1], g[2], 79, 30, 71.5, beta])
    R = [MT.por_degrau(*M.simula(fis, ctrl, var, SP, 200.0, 73.7, s, s0, 3.0)[:2], SP,
                       mediana=False) for s in sementes]
    return np.median(R, axis=0)


def J_gemeo(g, **kw):
    m = metricas_gemeo(g, **kw)
    return MT.J2(m) if CUSTO == "J2" else MT.J(m)


if __name__ == "__main__":
    for nome, g in COMPARAR.items():
        MT.imprime(metricas_gemeo(g), titulo="gemeo -- " + nome)

    lim = LIMITES if VARIANTE == 2 else LIMITES[:3]
    print(f"\nOtimizando (variante {VARIANTE}, custo {CUSTO})...")
    r = differential_evolution(J_gemeo, lim, maxiter=ITERACOES, popsize=15, seed=1,
                               polish=False, updating="deferred")
    g = r.x
    MT.imprime(metricas_gemeo(g), titulo=f"gemeo -- OTIMO {np.round(g, 3)}")
    np.save(os.path.join(E.PASTA_RES, "ganhos_otimos.npy"), g)

    # robustez
    x = np.load(PARAMETROS)
    variacoes = {
        "modelo usado": (FIS, S0),
        "atraso +2 amostras": (np.r_[FIS[:6], FIS[6] + 2, FIS[7:]], S0),
        "ruido do sensor 3,5 mm": (FIS, 3.5),
        "K x 1,3": (np.r_[FIS[0] * 1.3, FIS[1:]], S0),
    }
    outra = os.path.join(E.PASTA_PARAM, "fis_rodada2.npy")
    if os.path.exists(outra):
        variacoes["calibracao da rodada 2"] = M.carrega_fis(outra)
    print("\nSEMENTES NOVAS (20 realizacoes do ruido que a otimizacao nao viu)")
    for nc, gg in dict(COMPARAR, otimo=list(g)).items():
        m = metricas_gemeo(gg, sementes=range(100, 120))
        print(f"  {nc:32s} Mp {m[:, 2].min():3.0f}-{m[:, 2].max():3.0f}  J {MT.J(m):6.1f}  J2 {MT.J2(m):6.1f}")

    print("\nROBUSTEZ (Mp min-max | ef max | J | J2)")
    candidatos = dict(COMPARAR); candidatos["otimo"] = list(g)
    for nv, (f, s) in variacoes.items():
        for nc, gg in candidatos.items():
            m = metricas_gemeo(gg, fis=f, s0=s)
            print(f"  {nv:24s} {nc:32s} Mp {m[:, 2].min():3.0f}-{m[:, 2].max():3.0f}  "
                  f"ef {m[:, 5].max():4.0f}  J {MT.J(m):6.1f}  J2 {MT.J2(m):6.1f}")
