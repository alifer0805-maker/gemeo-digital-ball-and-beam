# -*- coding: utf-8 -*-
"""
PASSO 3 -- CONFERIR UM ENSAIO DA BANCADA CONTRA O GEMEO

Pega um CSV da bancada, simula no gemeo o mesmo firmware, os mesmos ganhos e
a mesma sequencia de setpoints, e mostra lado a lado: Mp, ts, tr, erro final,
batida do servo e a funcao custo J (bancada x gemeo). A razao J_bancada/J_gemeo
mede o quanto o gemeo e' otimista.

Para um ensaio novo: coloque o CSV em dados/ (ou o caminho completo) e preencha
ARQUIVO, VARIANTE e GANHOS.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("Agg")   # so gera arquivo, nao abre janela
import modelo as M
import metricas as MT
import ensaios as E

# configuracao
ENSAIO = "c6"              # chave de ensaios.py ... ou None para usar os campos abaixo
ARQUIVO = None             # ex.: r"C:\Users\...\ensaios_ballbeam\0810_101500_P0.26_I0.03_D0.35_B0.95.csv"
VARIANTE = 2               # 0 anterior | 1 v7 | 2 v8
GANHOS = [0.26, 0.03, 0.35, 0.95]          # Kp, Ki, Kd, beta
PARAMETROS = os.path.join(E.PASTA_PARAM, "fis_calibrado.npy")
SEMENTES = range(10)

if ENSAIO:
    d = E.carrega(ENSAIO); var, ctrl, desc = E.info(ENSAIO)
else:
    d = E.carrega(ARQUIVO); var = VARIANTE; desc = os.path.basename(ARQUIVO)
    b = GANHOS[3] if len(GANHOS) > 3 else 0.0
    ctrl = np.array([GANHOS[0], GANHOS[1], GANHOS[2], 79, 30, 71.5, b])

fis, s0 = M.carrega_fis(PARAMETROS)
sp = d.s.values.astype(float)
m_ban = MT.por_degrau(d.p.values, d.a.values, sp)
sims = [M.simula(fis, ctrl, var, sp, float(d.p[0]), float(d.a[0]), s, s0, 3.0) for s in SEMENTES]
m_gem = np.median([MT.por_degrau(y, u, sp) for y, u, _ in sims], axis=0)
MT.imprime(m_ban, m_gem, titulo=f"{desc}  |  gemeo: {os.path.basename(PARAMETROS)}")

t = np.arange(len(d)) * M.TS
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(t, sp, "k--", lw=1, label="setpoint")
ax[0].plot(t, d.p, lw=0.8, label="bancada")
ax[0].plot(t, sims[0][0], lw=1.0, label="gemeo")
ax[0].set_ylabel("posicao (mm)"); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title(desc)
ax[1].plot(t, d.a, lw=0.5, label="bancada")
ax[1].plot(t, sims[0][1], lw=0.5, label="gemeo")
ax[1].set_ylabel("servo (graus)"); ax[1].set_xlabel("tempo (s)"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout()
arq = os.path.join(E.PASTA_RES, f"avaliacao_{ENSAIO}.png")
plt.savefig(arq, dpi=200)
print("figura salva em", arq)
os.startfile(arq)   # abre no visualizador de imagens do Windows
