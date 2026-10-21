# -*- coding: utf-8 -*-
"""
PASSO 4 -- RESUMO DAS RODADAS E FIGURAS DO RELATORIO

  * Tabela de J por rodada: para cada rodada, o ensaio da bancada e o gemeo
    QUE FEZ A PREVISAO (calibrado antes do ensaio). Mostra a bancada
    melhorando e o quanto o gemeo ainda e' otimista.
  * Figura gemeo x bancada nos 6 ensaios usados na calibracao atual.
  * Figura do servo: v7 (derivada sem filtro) x v8 (derivada filtrada).
As figuras vao para resultados/.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import modelo as M
import metricas as MT
import ensaios as E

P = E.PASTA_PARAM
RODADAS = [  # (rodada, ensaio que conferiu a previsao, gemeo usado na previsao)
    ("1", "e5", "fis_rodada1.npy"),
    ("2", "e6", "fis_rodada2.npy"),
    ("3", "e7", "fis_rodada3.npy"),
    ("4", "e8", "fis_rodada3.npy"),
]

print("rodada  ensaio                               J bancada  J gemeo  razao")
for rod, ch, arq in RODADAS:
    d = E.carrega(ch); var, ctrl, desc = E.info(ch)
    fis, s0 = M.carrega_fis(os.path.join(P, arq))
    sp = d.s.values.astype(float)
    mb = MT.por_degrau(d.p.values, d.a.values, sp)
    mg = np.median([MT.por_degrau(*M.simula(fis, ctrl, var, sp, float(d.p[0]), float(d.a[0]),
                                            s, s0, 3.0)[:2], sp) for s in range(10)], axis=0)
    jb, jg = MT.J(mb), MT.J(mg)
    print(f"  {rod}     {desc:36s} {jb:7.0f}  {jg:7.0f}  {jb / jg:5.1f}")

# figura: gemeo x bancada (calibracao atual)
os.makedirs(E.PASTA_RES, exist_ok=True)
fis, s0 = M.carrega_fis(os.path.join(P, "fis_rodada3.npy"))
fig, axs = plt.subplots(3, 2, figsize=(11, 9), sharey=True)
for ax, ch in zip(axs.flat, ["e1", "e2", "e3", "e4", "e5", "e6"]):
    d = E.carrega(ch); var, ctrl, desc = E.info(ch)
    sp = d.s.values.astype(float); t = np.arange(len(d)) * M.TS
    ys = M.simula(fis, ctrl, var, sp, float(d.p[0]), float(d.a[0]), 3, s0, 3.0)[0]
    ax.plot(t, sp, "k--", lw=1, label="referencia")
    ax.plot(t, d.p, lw=0.6, label="bancada")
    ax.plot(t, ys, lw=1.0, label="gemeo")
    ax.set_title(f"{ch}: {desc}", loc="left", fontsize=9); ax.grid(alpha=.3)
axs[0, 0].legend(fontsize=8)
for ax in axs[-1]: ax.set_xlabel("tempo (s)")
for ax in axs[:, 0]: ax.set_ylabel("posicao (mm)")
plt.tight_layout(); plt.savefig(os.path.join(E.PASTA_RES, "gemeo_x_bancada.png"), dpi=150); plt.show()

# figura: servo v7 x v8
fig, axs = plt.subplots(2, 2, figsize=(11, 5.5), sharex=True, sharey="row")
for j, ch in enumerate(["e5", "e6"]):
    d = E.carrega(ch); _, _, desc = E.info(ch); t = np.arange(len(d)) * M.TS; m = t <= 100
    axs[0, j].plot(t[m], d.s[m], "k--", lw=1); axs[0, j].plot(t[m], d.p[m], lw=0.7)
    axs[1, j].plot(t[m], d.a[m], lw=0.35, color="0.3")
    du = np.mean(np.abs(np.diff(np.round(d.a.values))))
    axs[0, j].set_title(f"{desc}\nvariacao media do servo: {du:.1f} graus/amostra", loc="left", fontsize=9)
    axs[1, j].set_xlabel("tempo (s)")
    for a in axs[:, j]: a.grid(alpha=.3)
axs[0, 0].set_ylabel("posicao (mm)"); axs[1, 0].set_ylabel("servo (graus)")
plt.tight_layout(); plt.savefig(os.path.join(E.PASTA_RES, "servo_v7_x_v8.png"), dpi=150); plt.show()
