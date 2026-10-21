# -*- coding: utf-8 -*-
"""
Ensaio automatico do ball-and-beam (firmware v7 ou v8) -- para rodar no Spyder.

O que faz:
  1. abre a serial (o Arduino reinicia ao abrir; o script espera ele voltar);
  2. manda os ganhos (P, I, D) e o primeiro setpoint;
  3. percorre a lista de setpoints: so' troca quando a bola ESTABILIZOU
     (ficou dentro de +-TOL_MM por T_ESTAVEL s) ou quando estoura T_MAX s;
  4. grava TUDO que o Arduino manda num CSV, no mesmo formato de sempre
     (Channel 1..4 = posicao, angulo, setpoint, Ts), sem limite de tempo;
  5. no fim mostra o grafico e um resumo por degrau.

Ctrl+C (ou o botao vermelho do console do Spyder) interrompe e o CSV
e' salvo do mesmo jeito.

Precisa do pyserial:  pip install pyserial
(feche o Serial Monitor / Serial Plotter da IDE do Arduino antes de rodar)
"""
import time, csv, os
from datetime import datetime
import numpy as np
import serial
import matplotlib.pyplot as plt

# configuracao
PORTA = "COM4"            # Windows: "COM3", "COM4"... | Linux: "/dev/ttyUSB0"
BAUD = 2000000            # igual ao Serial.begin() do firmware

KP, KI, KD = 0.26, 0.01, 0.35
BETA_D = 0.95             # filtro da derivada (so' firmware v8); None = nao manda
LIMITE_SERVO = None       # o v7 nao tem comando pra isso; deixe None

SETPOINTS = [200, 80, 250, 200, 120, 250]   # mm, na ordem

T_ESPERA_INICIAL = 5.0    # s no 1o setpoint antes de comecar a contar
T_MIN = 6.0               # s minimo em cada setpoint
T_MAX = 25.0              # s maximo em cada setpoint (troca mesmo sem estabilizar)
TOL_MM = 8.0              # faixa considerada "estabilizado"
T_ESTAVEL = 3.0           # s seguidos dentro da faixa para trocar o setpoint
T_FINAL = 5.0             # s extras gravando depois do ultimo setpoint estabilizar

PASTA = os.path.join(os.path.expanduser("~"), "ensaios_ballbeam")
NOME = None               # None = nome automatico com data/hora e ganhos


def enviar(ser, cmd):
    """Manda um comando e espera o firmware processar.
    O firmware le' uma letra + parseFloat e depois descarta o resto do buffer,
    entao dois comandos colados se perdem -- por isso o intervalo."""
    ser.write((cmd + "\n").encode())
    ser.flush()
    time.sleep(0.15)
    print(f"  -> {cmd}")


def ler_linha(ser):
    """Devolve (pos, ang, sp, ts) ou None se a linha nao for de dados."""
    try:
        txt = ser.readline().decode(errors="ignore").strip()
        v = [float(x) for x in txt.split()]
        if len(v) >= 4:
            return v[:4]
    except ValueError:
        pass
    return None


def resumo(dados):
    d = np.array(dados)
    p, sp = d[:, 0], d[:, 2]
    ts = np.median(d[:, 3]) / 1000 if np.median(d[:, 3]) > 0 else 0.015
    g = np.r_[0, np.where(np.diff(sp) != 0)[0] + 1, len(sp)]
    print("\nDegrau        Mp(mm)   ts(s)   erro final(mm)")
    for i in range(1, len(g) - 1):
        a, b = g[i], g[i + 1]
        s, s0 = sp[a], sp[a - 1]
        yy = p[a:b]
        sg = np.sign(s - s0)
        Mp = max(0.0, np.max(sg * (yy - s)))
        fora = np.where(np.abs(yy - s) > 10)[0]
        t_s = (fora[-1] + 1) * ts if len(fora) else 0.0
        ef = np.mean(yy[-int(min(len(yy) / 3, 3 / ts)):]) - s
        print(f"{s0:4.0f} -> {s:4.0f}   {Mp:6.1f}   {t_s:5.1f}   {ef:+6.1f}")


def main():
    os.makedirs(PASTA, exist_ok=True)
    nome = NOME or datetime.now().strftime("%d%m_%H%M%S") + f"_P{KP}_I{KI}_D{KD}" + (f"_B{BETA_D}" if BETA_D is not None else "") + ".csv"
    caminho = os.path.join(PASTA, nome)
    dados = []

    ser = serial.Serial(PORTA, BAUD, timeout=0.1)
    try:
        print("Abrindo serial... (o Arduino reinicia)")
        time.sleep(2.5)
        ser.reset_input_buffer()

        print("Configurando:")
        enviar(ser, f"S{SETPOINTS[0]}")
        enviar(ser, f"P{KP}")
        enviar(ser, f"I{KI}")
        enviar(ser, f"D{KD}")
        if BETA_D is not None:
            enviar(ser, f"B{BETA_D}")
        enviar(ser, "A")                      # garante automatico, integral zerada
        ser.reset_input_buffer()

        idx = 0
        t_sp = time.time()
        t_dentro = None
        t_min_atual = T_ESPERA_INICIAL + T_MIN
        terminou_em = None
        print(f"\nSP {SETPOINTS[0]} mm")

        while True:
            v = ler_linha(ser)
            if v is None:
                continue
            dados.append(v)
            agora = time.time()
            pos = v[0]
            sp = SETPOINTS[idx]

            # estabilizou?
            if abs(pos - sp) <= TOL_MM:
                if t_dentro is None:
                    t_dentro = agora
            else:
                t_dentro = None
            estavel = t_dentro is not None and agora - t_dentro >= T_ESTAVEL
            decorrido = agora - t_sp

            if terminou_em is not None:
                if agora - terminou_em >= T_FINAL:
                    break
                continue

            if (decorrido >= t_min_atual and estavel) or decorrido >= T_MAX:
                motivo = "estabilizou" if estavel else "tempo maximo"
                print(f"   {motivo} em {decorrido:.1f} s (pos {pos:.1f} mm)")
                if idx == len(SETPOINTS) - 1:
                    terminou_em = agora
                    continue
                idx += 1
                enviar(ser, f"S{SETPOINTS[idx]}")
                print(f"SP {SETPOINTS[idx]} mm")
                t_sp = time.time(); t_dentro = None; t_min_atual = T_MIN

    except KeyboardInterrupt:
        print("\nInterrompido -- salvando o que foi gravado.")
    finally:
        ser.close()
        with open(caminho, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Channel 1", "Channel 2", "Channel 3", "Channel 4"])
            w.writerows(dados)
        print(f"\nCSV salvo: {caminho}  ({len(dados)} amostras, "
              f"{len(dados) * 0.015:.0f} s)")

    if len(dados) > 10:
        d = np.array(dados)
        t = np.arange(len(d)) * 0.015
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax[0].plot(t, d[:, 2], "k--", lw=1, label="setpoint")
        ax[0].plot(t, d[:, 0], lw=1, label="posicao")
        ax[0].set_ylabel("mm"); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[0].set_title(f"P{KP}  I{KI}  D{KD}  B{BETA_D}")
        ax[1].plot(t, d[:, 1], lw=.6); ax[1].set_ylabel("servo (graus)")
        ax[1].set_xlabel("tempo (s)"); ax[1].grid(alpha=.3)
        plt.tight_layout(); plt.show()
        resumo(dados)


if __name__ == "__main__":
    main()
