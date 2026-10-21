# -*- coding: utf-8 -*-
"""
Metricas de desempenho e a FUNCAO CUSTO J.

A mesma funcao serve para a bancada e para o gemeo -- e' isso que permite
comparar os dois com o mesmo numero.

Por degrau de setpoint:
  Mp   sobressinal [mm]: quanto a bola passa do alvo no sentido do movimento
  ts   tempo de acomodacao [s]: ultimo instante fora da faixa +-10 mm
  tr   tempo de subida [s]: de 10% a 90% do degrau
  ef   erro final [mm]: |media dos ultimos 3 s - setpoint|
  du   "batida" do servo: variacao media do comando, graus por amostra
  sat  % do tempo com o servo no limite

J = soma sobre os degraus de ( Mp/10 + ts + 0.3*ef + 1.5*du )
    (pesos: 10 mm de sobressinal "custam" o mesmo que 1 s de acomodacao)
"""
import numpy as np
from scipy.ndimage import median_filter

TS = 0.015
FAIXA_TS = 10.0      # mm
PESOS = dict(Mp=1 / 10, ts=1.0, ef=1.5 / 5, du=1.5)
NOMES = ["sp_ant", "sp", "Mp", "ts", "tr", "ef", "du", "sat", "iae"]


def por_degrau(y, u, sp, mediana=True, n_min=200):
    """Metricas de cada degrau. Retorna array (n_degraus x 9), colunas em NOMES.
    mediana=True passa a posicao por um filtro de mediana de 15 amostras para
    que picos isolados de leitura (bancada) nao contem como sobressinal."""
    y = np.asarray(y, float); u = np.asarray(u, float); sp = np.asarray(sp, float)
    if mediana:
        y = median_filter(y, 15, mode="nearest")
    g = np.r_[0, np.where(np.diff(sp) != 0)[0] + 1, len(sp)]
    out = []
    for i in range(1, len(g) - 1):
        a, b = g[i], g[i + 1]
        if b - a < n_min:                       # degrau curto demais: ignora
            continue
        s, s0 = sp[a], sp[a - 1]
        yy, uu = y[a:b], u[a:b]
        sg = np.sign(s - s0)
        Mp = max(0.0, np.max(sg * (yy - s)))
        fora = np.where(np.abs(yy - s) > FAIXA_TS)[0]
        ts = min((fora[-1] + 1) * TS if len(fora) else 0.0, 15.0)
        lo, hi = s0 + 0.1 * (s - s0), s0 + 0.9 * (s - s0)
        tr = (np.argmax(sg * (yy - hi) >= 0) - np.argmax(sg * (yy - lo) >= 0)) * TS
        ef = abs(np.mean(yy[-200:]) - s)
        du = np.mean(np.abs(np.diff(np.round(uu))))
        nivel = 79 + (71.5 - 79) * (s - 80) / 170
        sat = (np.abs(uu - nivel) > 29.5).mean() * 100
        iae = np.mean(np.abs(yy[:int(15 / TS)] - s))   # erro medio absoluto nos 15 s do degrau [mm]
        out.append((s0, s, Mp, ts, tr, ef, du, sat, iae))
    return np.array(out)


def J(m):
    """Funcao custo versao 1 (rodadas 1-4 do relatorio)."""
    return float(np.sum(PESOS["Mp"] * m[:, 2] + PESOS["ts"] * m[:, 3]
                        + PESOS["ef"] * m[:, 5] + PESOS["du"] * m[:, 6]))


PESOS2 = dict(Mp=1 / 10, iae=1 / 3, ef=1.5 / 5, du=1.5)


def J2(m):
    """Funcao custo versao 2: troca ts (que satura em 15 s quando a bola para
    por atrito fora da faixa de +-10 mm) pelo erro medio absoluto do degrau.
    J2 = soma( Mp/10 + IAE/3 + 0.3*ef + 1.5*du ),  IAE em mm (media nos 15 s)"""
    return float(np.sum(PESOS2["Mp"] * m[:, 2] + PESOS2["iae"] * m[:, 8]
                        + PESOS2["ef"] * m[:, 5] + PESOS2["du"] * m[:, 6]))


def imprime(m_ban, m_gem=None, titulo=""):
    """Tabela lado a lado bancada / gemeo."""
    if titulo:
        print("\n== " + titulo)
    if m_gem is None:
        print("degrau        Mp(mm)  ts(s)  tr(s)  ef(mm)  du(o/am)  sat(%)")
        for r in m_ban:
            print(f"{r[0]:4.0f}->{r[1]:4.0f}  {r[2]:7.1f} {r[3]:6.1f} {r[4]:6.2f} {r[5]:7.1f} {r[6]:8.2f} {r[7]:6.0f}")
        print(f"J = {J(m_ban):.1f}   J2 = {J2(m_ban):.1f}   (IAE por degrau: {np.round(m_ban[:, 8], 1)})")
        return
    print("degrau        Mp ban/gem    ts ban/gem    tr ban/gem    ef ban/gem   du ban/gem")
    for a, b in zip(m_ban, m_gem):
        print(f"{a[0]:4.0f}->{a[1]:4.0f}  {a[2]:5.1f}/{b[2]:5.1f}  {a[3]:5.1f}/{b[3]:5.1f}  "
              f"{a[4]:5.2f}/{b[4]:5.2f}  {a[5]:5.1f}/{b[5]:5.1f}  {a[6]:4.2f}/{b[6]:4.2f}")
    for nome, f in (("J ", J), ("J2", J2)):
        jb, jg = f(m_ban), f(m_gem)
        print(f"{nome} bancada = {jb:.1f}   {nome} gemeo = {jg:.1f}   razao = {jb / jg:.2f}")
