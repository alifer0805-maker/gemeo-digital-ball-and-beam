# -*- coding: utf-8 -*-
"""
PASSO 5 -- ENSAIO COM GANHOS VARIAVEIS (o gemeo escolhe os ganhos a cada setpoint)

Igual ao ensaio_ballbeam.py, mas antes de cada novo setpoint:
  1. le' a posicao atual da bola;
  2. otimiza no gemeo os ganhos para ir dali ate' o setpoint novo (~0,5 s);
  3. manda P, I, D, B e so' depois o S.

Com ADAPTATIVO = False roda o MESMO roteiro com ganhos fixos -- rode os dois e
compare o J (o resumo no fim ja' imprime). Firmware: v8.

Saidas (em ~/ensaios_ballbeam):
  ..._adaptativo.csv  /  ..._fixo.csv   -> mesmo formato de sempre (4 canais)
  ..._ganhos.csv                         -> ganhos usados em cada degrau + J previsto
"""
import os, time, csv
from datetime import datetime
import numpy as np
import serial
import matplotlib.pyplot as plt
import modelo as M
import metricas as MT
import ganhos_por_setpoint as G
import visual as V

# configuracao
PORTA = "COM4"
BAUD = 2000000
ADAPTATIVO = True                       # False = ganhos fixos abaixo (referencia)
GANHOS_FIXOS = [0.26, 0.01, 0.35, 0.95]  # Kp, Ki, Kd, beta (tambem usados no 1o setpoint se fixo)
VISUAL = True                           # janela ao vivo com a bola real e a bola do gemeo
PARAMETROS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parametros", "fis_rodada3.npy")

SETPOINTS = [200, 80, 250, 200, 120, 250]
T_ESPERA_INICIAL = 5.0
T_MIN, T_MAX = 6.0, 25.0
TOL_MM, T_ESTAVEL = 8.0, 3.0
T_FINAL = 5.0
PASTA = os.path.join(os.path.expanduser("~"), "ensaios_ballbeam")


def enviar(ser, cmd, mostrar=True):
    ser.write((cmd + "\n").encode()); ser.flush()
    time.sleep(0.15)                     # o firmware descarta comandos colados
    if mostrar:
        print(f"  -> {cmd}")


def ler_linha(ser):
    try:
        v = [float(x) for x in ser.readline().decode(errors="ignore").split()]
        if len(v) >= 4:
            return v[:4]
    except ValueError:
        pass
    return None


def aplica_ganhos(ser, g):
    enviar(ser, f"P{g[0]:.3f}", False); enviar(ser, f"I{g[1]:.4f}", False)
    enviar(ser, f"D{g[2]:.3f}", False); enviar(ser, f"B{g[3]:.3f}", False)


def previsao(pos, ang, sp, g, fis, s0):
    """Trajetoria que o gemeo preve para este degrau com os ganhos g (bola fantasma)."""
    spv = np.full(int(15 / M.TS), float(sp))
    ct = np.array([g[0], g[1], g[2], 79, 30, 71.5, g[3]])
    return M.simula(fis, ct, 2, spv, float(pos), float(ang), 0, s0, 3.0)[0]


def novo_setpoint(ser, pos, sp, fis, s0, log, ang=None, vis=None):
    """Escolhe os ganhos (gemeo ou fixos), manda, e manda o setpoint."""
    if ADAPTATIVO:
        t0 = time.time()
        g, jprev = G.otimiza_degrau(pos, sp, fis, s0)
        dt = time.time() - t0
        print(f"SP {sp} mm  (bola em {pos:.0f})  gemeo: P{g[0]:.3f} I{g[1]:.4f} D{g[2]:.3f} "
              f"B{g[3]:.3f}  J previsto {jprev:.1f}  [{dt:.2f} s]")
    else:
        g, jprev = np.array(GANHOS_FIXOS), np.nan
        print(f"SP {sp} mm  (ganhos fixos)")
    aplica_ganhos(ser, g)
    enviar(ser, f"S{sp}", False)
    if vis is not None:
        vis.nova_previsao(previsao(pos, ang if ang is not None else G.nivel(pos), sp, g, fis, s0))
        vis.ganhos = g
    log.append([pos, sp, *np.round(g, 4), round(jprev, 2)])


def main():
    fis, s0 = M.carrega_fis(PARAMETROS)
    vis = None
    if VISUAL:
        V.usa_janela_externa()
        vis = V.Visualizador("ball-and-beam -- " + ("ganhos do gemeo" if ADAPTATIVO else "ganhos fixos"))
        vis.ganhos = np.array(GANHOS_FIXOS)
    if ADAPTATIVO:
        print("Compilando o gemeo (so' na 1a vez, ~15 s)...")
        G.aquece(fis, s0)

    os.makedirs(PASTA, exist_ok=True)
    base = datetime.now().strftime("%d%m_%H%M%S") + ("_adaptativo" if ADAPTATIVO else "_fixo")
    dados, log = [], []
    ser = serial.Serial(PORTA, BAUD, timeout=0.1)
    try:
        print("Abrindo serial... (o Arduino reinicia)")
        time.sleep(2.5)
        ser.reset_input_buffer()
        aplica_ganhos(ser, GANHOS_FIXOS)
        enviar(ser, f"S{SETPOINTS[0]}")
        enviar(ser, "A")
        ser.reset_input_buffer()

        idx, t_sp, t_dentro, t_min, fim = 0, time.time(), None, T_ESPERA_INICIAL + T_MIN, None
        print(f"SP {SETPOINTS[0]} mm (inicial, ganhos fixos)")
        while True:
            v = ler_linha(ser)
            if v is None:
                continue
            dados.append(v)
            agora, pos, sp = time.time(), v[0], SETPOINTS[idx]
            if vis is not None:
                g = vis.ganhos
                vis.atualiza(pos, v[1], sp, f"P{g[0]:.3f}  I{g[1]:.4f}  D{g[2]:.3f}  B{g[3]:.3f}")
            t_dentro = (t_dentro or agora) if abs(pos - sp) <= TOL_MM else None
            estavel = t_dentro is not None and agora - t_dentro >= T_ESTAVEL
            if fim is not None:
                if agora - fim >= T_FINAL:
                    break
                continue
            if (agora - t_sp >= t_min and estavel) or agora - t_sp >= T_MAX:
                print(f"   {'estabilizou' if estavel else 'tempo maximo'} em {agora - t_sp:.1f} s")
                if idx == len(SETPOINTS) - 1:
                    fim = agora
                    continue
                idx += 1
                novo_setpoint(ser, pos, SETPOINTS[idx], fis, s0, log, ang=v[1], vis=vis)
                t_sp, t_dentro, t_min = time.time(), None, T_MIN
    except KeyboardInterrupt:
        print("\nInterrompido -- salvando.")
    finally:
        ser.close()
        with open(os.path.join(PASTA, base + ".csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["Channel 1", "Channel 2", "Channel 3", "Channel 4"]); w.writerows(dados)
        with open(os.path.join(PASTA, base + "_ganhos.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["pos_inicial", "setpoint", "Kp", "Ki", "Kd", "beta", "J_previsto"]); w.writerows(log)
        print(f"\nSalvo: {base}.csv e {base}_ganhos.csv em {PASTA}")

    if len(dados) > 300:
        d = np.array(dados)
        m = MT.por_degrau(d[:, 0], d[:, 1], d[:, 2])
        MT.imprime(m, titulo="adaptativo" if ADAPTATIVO else "ganhos fixos")
        t = np.arange(len(d)) * M.TS
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax[0].plot(t, d[:, 2], "k--", lw=1); ax[0].plot(t, d[:, 0], lw=1)
        ax[0].set_ylabel("posicao (mm)"); ax[0].grid(alpha=.3)
        ax[0].set_title(("ganhos variaveis (gemeo)" if ADAPTATIVO else "ganhos fixos") + f"   J = {MT.J(m):.1f}")
        ax[1].plot(t, d[:, 1], lw=.5); ax[1].set_ylabel("servo (graus)"); ax[1].set_xlabel("tempo (s)"); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
