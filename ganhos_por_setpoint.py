# -*- coding: utf-8 -*-
"""
Otimizacao RAPIDA de ganhos para UM degrau (~0,5 s).

Dado onde a bola esta agora e o setpoint novo, simula no gemeo o firmware v8
indo de um para o outro e procura Kp, Ki, Kd, beta que minimizam J daquele
degrau. E' isso que roda a cada mudanca de setpoint no ensaio adaptativo.
"""
import numpy as np
from scipy.optimize import differential_evolution
import modelo as M
import metricas as MT

LIMITES = [(0.05, 0.6), (0.0, 0.08), (0.0, 0.8), (0.8, 0.97)]   # Kp, Ki, Kd, beta
DURACAO = 12.0         # s simulados por degrau (o atrito aparece no fim)
SEMENTES = range(4)    # realizacoes do ruido do sensor


def nivel(sp):
    return 79 + (71.5 - 79) * (sp - 80) / 170


def otimiza_degrau(p_atual, sp_novo, fis, s0, iteracoes=15, semente=0):
    """Retorna (ganhos [Kp, Ki, Kd, beta], J previsto)."""
    sp = np.r_[float(p_atual), np.full(int(DURACAO / M.TS), float(sp_novo))]
    ang0 = nivel(p_atual)

    def J(g):
        ct = np.array([g[0], g[1], g[2], 79, 30, 71.5, g[3]])
        tot = 0.0
        for s in SEMENTES:
            y, u, _ = M.simula(fis, ct, 2, sp, float(p_atual), ang0, s, s0, 3.0)
            tot += MT.J2(MT.por_degrau(y, u, sp, mediana=False, n_min=100))
        return tot / len(SEMENTES)

    r = differential_evolution(J, LIMITES, maxiter=iteracoes, popsize=10, seed=semente,
                               polish=False, updating="deferred")
    return r.x, r.fun


def aquece(fis, s0):
    """Compila o numba antes de abrir a serial (a 1a chamada demora ~15 s)."""
    otimiza_degrau(200, 80, fis, s0, iteracoes=1)
