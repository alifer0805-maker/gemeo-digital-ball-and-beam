# -*- coding: utf-8 -*-
"""
Janela ao vivo do ball-and-beam: a viga inclinando, a bola real, o setpoint e
a "bola fantasma" do gemeo (o que o gemeo previu para aquele degrau), mais um
grafico das ultimas dezenas de segundos.

No Spyder a janela precisa abrir FORA do console:
  Ferramentas > Preferencias > Console IPython > Graficos > Backend: Automatico
  (ou rode  %matplotlib qt  no console antes). O script tenta trocar sozinho.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

COMPRIMENTO = 320.0    # mm (viga)
RAIO = 15.0            # mm (bola, so' para o desenho)
EXAGERO = 0.5          # o desenho inclina EXAGERO x (angulo do servo - nivel)
JANELA_S = 30.0        # s mostrados no grafico de baixo
FPS = 12


def usa_janela_externa():
    """No Spyder/IPython troca para o backend Qt (janela separada)."""
    try:
        ip = get_ipython()            # noqa: F821  (existe dentro do Spyder)
        ip.run_line_magic("matplotlib", "qt")
    except Exception:
        pass


def _nivel(sp):
    return 79 + (71.5 - 79) * (sp - 80) / 170


class Visualizador:
    def __init__(self, titulo="ball-and-beam"):
        plt.ion()
        self.fig, (self.ax, self.ax2) = plt.subplots(
            2, 1, figsize=(9, 6.5), gridspec_kw=dict(height_ratios=[1.5, 1]))
        try:
            self.fig.canvas.manager.set_window_title(titulo)
        except Exception:
            pass
        ax = self.ax
        ax.set_xlim(-30, COMPRIMENTO + 30); ax.set_ylim(-110, 110)
        ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((-28, -20), 22, 40, color="0.35"))          # sensor
        ax.text(-17, -38, "sensor", ha="center", fontsize=8, color="0.35")
        ax.plot([COMPRIMENTO / 2, COMPRIMENTO / 2], [-70, -8], lw=4, color="0.6")   # suporte
        ax.plot([COMPRIMENTO / 2], [-6], marker="o", ms=8, color="0.4")          # pivo
        (self.viga,) = ax.plot([], [], lw=6, color="#4a4a4a", solid_capstyle="round")
        (self.alvo,) = ax.plot([], [], marker="v", ms=11, color="#2f9e44", ls="none")
        self.fantasma = Circle((0, 0), RAIO, fc="none", ec="#eb6834", lw=2, ls="--", alpha=0.9)
        self.bola = Circle((0, 0), RAIO, fc="#2a78d6", ec="#1b4f8f", lw=1.5)
        ax.add_patch(self.fantasma); ax.add_patch(self.bola)
        self.txt = ax.text(0, 95, "", fontsize=9, family="monospace", va="top")
        ax.text(COMPRIMENTO + 25, 95, "azul: bola real\nlaranja tracejada: gemeo\nverde: setpoint",
                ha="right", va="top", fontsize=8, color="0.3")
        a2 = self.ax2
        (self.l_sp,) = a2.plot([], [], "--", color="#2f9e44", lw=1.2, label="setpoint", drawstyle="steps-post")
        (self.l_g,) = a2.plot([], [], color="#eb6834", lw=1.2, label="gemeo")
        (self.l_p,) = a2.plot([], [], color="#2a78d6", lw=1.2, label="bancada")
        a2.set_ylim(20, 320); a2.set_xlim(0, JANELA_S); a2.grid(alpha=.3)
        a2.set_xlabel("tempo (s)"); a2.set_ylabel("posicao (mm)"); a2.legend(loc="upper left", fontsize=8, ncol=3)
        self.t, self.p, self.s, self.g = [], [], [], []
        self.t0 = time.time(); self.ultimo = 0.0
        self.prev = None; self.prev_t0 = None
        self.fig.tight_layout(); plt.show(block=False)

    def nova_previsao(self, y_prev):
        """Trajetoria prevista pelo gemeo para o degrau que comecou agora."""
        self.prev = np.asarray(y_prev); self.prev_t0 = time.time()

    def _fantasma_agora(self):
        if self.prev is None:
            return np.nan
        k = int((time.time() - self.prev_t0) / 0.015)
        return float(self.prev[min(k, len(self.prev) - 1)])

    def atualiza(self, pos, ang, sp, extra=""):
        agora = time.time()
        g = self._fantasma_agora()
        self.t.append(agora - self.t0); self.p.append(pos); self.s.append(sp); self.g.append(g)
        if agora - self.ultimo < 1.0 / FPS:
            return
        self.ultimo = agora
        inc = np.radians((ang - _nivel(sp)) * EXAGERO)       # angulo < nivel => direita desce
        c, s_ = np.cos(inc), np.sin(inc)
        meio = COMPRIMENTO / 2
        def ponto(x, h=0.0):                                   # x ao longo da viga, h acima dela
            dx = x - meio
            return meio + dx * c - h * s_, dx * s_ + h * c
        x0, y0 = ponto(0); x1, y1 = ponto(COMPRIMENTO)
        self.viga.set_data([x0, x1], [y0, y1])
        self.bola.center = ponto(np.clip(pos, 0, COMPRIMENTO), RAIO + 4)
        if np.isfinite(g):
            self.fantasma.center = ponto(np.clip(g, 0, COMPRIMENTO), RAIO + 4); self.fantasma.set_visible(True)
        else:
            self.fantasma.set_visible(False)
        self.alvo.set_data(*[[v] for v in ponto(sp, 2 * RAIO + 14)])
        self.txt.set_text(f"posicao {pos:6.1f} mm   setpoint {sp:5.0f} mm   servo {ang:5.1f} graus\n{extra}")
        tt = np.array(self.t); m = tt >= tt[-1] - JANELA_S
        base = max(0.0, tt[-1] - JANELA_S)
        self.l_p.set_data(tt[m] - base, np.array(self.p)[m])
        self.l_sp.set_data(tt[m] - base, np.array(self.s)[m])
        self.l_g.set_data(tt[m] - base, np.array(self.g)[m])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
