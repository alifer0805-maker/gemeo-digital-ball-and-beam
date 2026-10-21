# -*- coding: utf-8 -*-
"""
Os ensaios da bancada (07/10/2026) e como carrega-los.

Cada ensaio diz QUAL firmware rodou e COM QUAIS ganhos -- o gemeo simula
exatamente aquilo. ctrl = [Kp, Ki, Kd, N1, LIM, N2, beta] (ver modelo.simula).
"""
import os
import numpy as np
import pandas as pd

PASTA = os.path.dirname(os.path.abspath(__file__))
PASTA_DADOS = os.path.join(PASTA, "dados")
PASTA_PARAM = os.path.join(PASTA, "parametros")
PASTA_RES = os.path.join(PASTA, "resultados")

# nome do arquivo, variante do firmware, ctrl, descricao
ENSAIOS = {
    "e1": ("e1_codigo_anterior_N80.csv",           0, [0.25, 0.005, 0.15, 80, 40, 0, 0],      "codigo anterior, nivel 80, +-40"),
    "e2": ("e2_codigo_anterior_N75.csv",           0, [0.25, 0.005, 0.15, 75, 30, 0, 0],      "codigo anterior, nivel 75, +-30"),
    "e3": ("e3_v7_P0.25_I0.005_D0.15.csv",         1, [0.25, 0.005, 0.15, 79, 30, 71.5, 0],   "v7 (0,25/0,005/0,15)"),
    "e4": ("e4_v7_P0.25_I0.005_D0.15_b.csv",       1, [0.25, 0.005, 0.15, 79, 30, 71.5, 0],   "v7 (0,25/0,005/0,15)"),
    "e5": ("e5_v7_P0.39_I0.03_D0.38.csv",          1, [0.39, 0.03, 0.38, 79, 30, 71.5, 0],    "v7 (0,39/0,03/0,38) -- rodada 1"),
    "e6": ("e6_v8_P0.42_I0.08_D0.36_B0.95.csv",    2, [0.42, 0.08, 0.36, 79, 30, 71.5, 0.95], "v8 (0,42/0,08/0,36) -- rodada 2"),
    "e7": ("e7_v8_P0.26_I0.01_D0.35_B0.95.csv",    2, [0.26, 0.01, 0.35, 79, 30, 71.5, 0.95], "v8 (0,26/0,01/0,35) -- rodada 3"),
    "e8": ("e8_v8_P0.26_I0.03_D0.35_B0.95.csv",    2, [0.26, 0.03, 0.35, 79, 30, 71.5, 0.95], "v8 (0,26/0,03/0,35) -- rodada 4"),
    "e9": ("e9_v8_P0.26_I0.01_D0.35_B0.95_fixo.csv", 2, [0.26, 0.01, 0.35, 79, 30, 71.5, 0.95], "v8 (0,26/0,01/0,35) -- ref. fixa do adaptativo"),
    "c1": ("c1.csv", 2, [0.25, 0.005, 0.15, 79, 30, 71.5, 0.0], "v8 B0 (0,25/0,005/0,15)"),
    "c2": ("c2.csv", 2, [0.25, 0.005, 0.15, 79, 30, 71.5, 0.0], "v8 B0 (0,25/0,005/0,15)"),
    "c3": ("c3.csv", 2, [0.39, 0.03, 0.38, 79, 30, 71.5, 0.0], "v8 B0 (0,39/0,03/0,38)"),
    "c4": ("c4.csv", 2, [0.42, 0.08, 0.36, 79, 30, 71.5, 0.95], "v8 (0,42/0,08/0,36)"),
    "c5": ("c5.csv", 2, [0.26, 0.01, 0.35, 79, 30, 71.5, 0.95], "v8 (0,26/0,01/0,35) validacao"),
    "c6": ("c6.csv", 2, [0.26, 0.01, 0.35, 79, 30, 71.5, 0.95], "v8 (0,26/0,01/0,35) validacao"),
}



def carrega(chave_ou_arquivo):
    """Le um CSV da bancada (formato do Serial Plotter / do script do Spyder).
    Aceita a chave ('e5') ou um caminho de arquivo.
    Retorna DataFrame com colunas p (posicao), a (angulo), s (setpoint), dt (Ts real)."""
    if chave_ou_arquivo in ENSAIOS:
        caminho = os.path.join(PASTA_DADOS, ENSAIOS[chave_ou_arquivo][0])
    else:
        caminho = chave_ou_arquivo
    d = pd.read_csv(caminho)
    d.columns = ["p", "a", "s", "dt"][:d.shape[1]]
    return d


def info(chave):
    """(variante, ctrl como array, descricao) de um ensaio."""
    _, var, ctrl, desc = ENSAIOS[chave]
    return var, np.array(ctrl, float), desc
