# -*- coding: utf-8 -*-
"""
PASSO 1 -- CALIBRACAO DO GEMEO (identificacao caixa cinza em malha fechada)

Ideia: para um conjunto candidato de parametros fisicos, simula CADA ensaio da
bancada com o MESMO firmware e os MESMOS ganhos que rodaram nele, e mede o
quanto a simulacao se afasta da bancada. A evolucao diferencial procura os
parametros que minimizam essa distancia.

Custo da calibracao (por ensaio, somado):
  1.5 x erro medio de posicao nos 4 s apos cada degrau  (forma do transitorio)
  0.7 x erro medio de posicao no ensaio inteiro         (regime, bola presa)
  2.0 x |batida do servo simulada - batida medida|      (ruido -> derivada)

As simulacoes de cada avaliacao rodam em paralelo (modelo.simula_lote), uma
por nucleo do processador. O melhor resultado e' salvo a cada geracao em SAIDA,
entao da' pra parar com Ctrl+C quando o custo parar de cair.
"""
import os, time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.ndimage import median_filter
import modelo as M
import ensaios as E

# configuracao
USAR = ["c1", "c2", "c3", "c4", "c5", "c6"]   # rodada 3 usou e1-e6; validacao usou c1-c4
ITERACOES = 60
POPULACAO = 20
PARTIDA = os.path.join(E.PASTA_PARAM, "fis_calibrado.npy")   # chute inicial (None = sem)
PESO_ERRO_FINAL = 1.5            # peso do erro de regime por degrau (0 = como na rodada 3)
SEMENTES_RUIDO = (0, 1, 2)                     # realizacoes do ruido por avaliacao
SAIDA = os.path.join(E.PASTA_PARAM, "fis_rodada5.npy")
# limites de busca: K, a_s, a_c, b, c0, c1, D, kv, s0, omega
LIMITES = [(300, 3000), (20, 600), (5, 300), (0.05, 3), (72, 79), (-0.08, 0.0),
           (1, 16), (0, 10), (0.5, 4), (60, 400)]
NOMES = ["K", "a_s", "a_c", "b", "c0", "c1", "D", "kv", "s0", "omega"]

DADOS = []
for chave in USAR:
    d = E.carrega(chave)
    var, ctrl, _ = E.info(chave)
    sp = d.s.values.astype(float)
    degraus = [k for k in np.where(np.diff(sp) != 0)[0] + 1 if len(sp) - k > 270]
    batida = np.mean(np.abs(np.diff(np.round(d.a.values))))
    y = median_filter(d.p.values, 15, mode="nearest")
    DADOS.append(dict(var=var, ctrl=ctrl, sp=sp, y=y, p0=float(d.p[0]),
                      a0=float(d.a[0]), degraus=degraus, batida=batida))


# tudo emendado num vetor so', pro simula_lote rodar em paralelo
_ini = np.cumsum([0] + [len(e["sp"]) for e in DADOS])
LOTE = dict(ctrls=np.array([e["ctrl"] for e in DADOS]),
            variantes=np.array([e["var"] for e in DADOS], dtype=np.int64),
            sp=np.concatenate([e["sp"] for e in DADOS]),
            ini=_ini[:-1].astype(np.int64), fim=_ini[1:].astype(np.int64),
            p0=np.array([e["p0"] for e in DADOS]), a0=np.array([e["a0"] for e in DADOS]),
            sem=np.array(SEMENTES_RUIDO, dtype=np.int64))


def custo_calibracao(x):
    K, a_s, a_c, b, c0, c1, D, kv, s0, om = x
    fis = np.array([K, a_s, a_c, b, c0, c1, round(D), kv, 0.0, om])
    L = LOTE
    YY, UU = M.simula_lote(fis, L["ctrls"], L["variantes"], L["sp"], L["ini"], L["fim"],
                           L["p0"], L["a0"], L["sem"], s0, 3.0)
    tot = 0.0; n = 0
    for i, e in enumerate(DADOS):
        a, z = L["ini"][i], L["fim"][i]
        Y = np.median(YY[:, a:z], axis=0)
        batida_sim = np.mean([np.mean(np.abs(np.diff(np.round(u[a:z])))) for u in UU])
        gs = e["degraus"]
        tot += 2.0 * len(gs) * abs(batida_sim - e["batida"])
        fins = list(gs[1:]) + [len(e["sp"])]
        for k, kf in zip(gs, fins):
            w = slice(k, k + 270)                       # 4 s apos o degrau
            tot += 1.5 * np.mean(np.abs(Y[w] - e["y"][w])); n += 1
            r = slice(max(k, kf - 200), kf)             # ultimos 3 s do degrau: onde a bola parou
            tot += PESO_ERRO_FINAL * abs(np.mean(Y[r]) - np.mean(e["y"][r]))
        tot += 0.7 * len(gs) * np.mean(np.abs(Y[gs[0]:] - e["y"][gs[0]:]))
    return tot / n


T0 = time.time()


def salva_parcial(xk, convergence=None):
    np.save(SAIDA, xk)   # guarda o melhor ate agora (pode parar com Ctrl+C)
    print(f"   {time.time() - T0:.0f} s")


if __name__ == "__main__":
    print(f"Calibrando com {USAR} ({ITERACOES} iteracoes)...")
    t0 = T0 = time.time()
    x0 = np.clip(np.load(PARTIDA), [l[0] for l in LIMITES], [l[1] for l in LIMITES]) if PARTIDA else None
    if x0 is not None:
        print(f"custo da partida: {custo_calibracao(x0):.2f}")
    r = differential_evolution(custo_calibracao, LIMITES, maxiter=ITERACOES,
                               popsize=POPULACAO, seed=0, tol=1e-3, polish=False,
                               updating="deferred", disp=True, x0=x0,
                               callback=salva_parcial)
    print(f"\n{time.time() - t0:.0f} s   custo final = {r.fun:.2f}")
    for nome, v in zip(NOMES, r.x):
        print(f"  {nome:6s} = {v:.4g}")
    print(f"  atraso do servo = {round(r.x[6])} amostras = {round(r.x[6]) * 15} ms")
    np.save(SAIDA, r.x)
    print("salvo em", SAIDA)
