# -*- coding: utf-8 -*-
"""
Roda em sequencia os ensaios da calibracao (lista ENSAIOS_CAL) e salva
cada CSV direto em dados/. Usa o mesmo roteiro e as mesmas funcoes do
ensaio_ballbeam.py.
No fim imprime as linhas pra colar no ENSAIOS do ensaios.py.
"""
import os, csv, time
import numpy as np
import serial
import matplotlib.pyplot as plt
from ensaio_ballbeam import (enviar, ler_linha, resumo, BAUD, SETPOINTS,
                             T_ESPERA_INICIAL, T_MIN, T_MAX, TOL_MM,
                             T_ESTAVEL, T_FINAL)

PORTA = "COM4"
PASTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados")

# nome, Kp, Ki, Kd, beta, descricao
ENSAIOS_CAL = [
    ("d7", 0.26, 0.01, 0.35, 0.95, "rodada 3"),
    ("d8", 0.26, 0.01, 0.35, 0.95, "rodada 3"),
]

T_ENTRE = 8.0          # s com ganhos da rodada 3 em 200mm antes de cada ensaio
T_DESCANSO = 180.0     # s com o servo parado entre ensaios (esfriar)
ANG_DESCANSO = 74      # angulo do servo no descanso (perto do nivel)
PAUSA = False          # True = espera Enter entre ensaios


def descansa(ser, seg):
    # servo parado em manual: sem PID nao fica corrigindo e esquenta menos
    enviar(ser, f"M{ANG_DESCANSO}")
    print(f"descanso de {seg:.0f} s (servo parado)")
    t0 = time.time()
    while time.time() - t0 < seg:
        ler_linha(ser)


def roda_um(ser, kp, ki, kd, beta):
    # leva a bola pra 200 com um controlador estavel
    enviar(ser, f"S{SETPOINTS[0]}")
    enviar(ser, "P0.26"); enviar(ser, "I0.01"); enviar(ser, "D0.35"); enviar(ser, "B0.95")
    enviar(ser, "A")
    t0 = time.time()
    while time.time() - t0 < T_ENTRE:
        ler_linha(ser)

    enviar(ser, f"P{kp}"); enviar(ser, f"I{ki}"); enviar(ser, f"D{kd}"); enviar(ser, f"B{beta}")
    enviar(ser, "A")
    ser.reset_input_buffer()

    dados = []
    idx = 0
    t_sp = time.time(); t_dentro = None
    t_min_atual = T_ESPERA_INICIAL + T_MIN
    terminou_em = None
    print(f"SP {SETPOINTS[0]} mm")
    while True:
        v = ler_linha(ser)
        if v is None:
            continue
        dados.append(v)
        agora = time.time()
        pos = v[0]
        sp = SETPOINTS[idx]
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
            print(f"   {'estabilizou' if estavel else 'tempo maximo'} em {decorrido:.1f} s (pos {pos:.1f} mm)")
            if idx == len(SETPOINTS) - 1:
                terminou_em = agora
                continue
            idx += 1
            enviar(ser, f"S{SETPOINTS[idx]}")
            print(f"SP {SETPOINTS[idx]} mm")
            t_sp = time.time(); t_dentro = None; t_min_atual = T_MIN
    return dados


def salva(dados, caminho):
    with open(caminho, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Channel 1", "Channel 2", "Channel 3", "Channel 4"])
        w.writerows(dados)


def main():
    os.makedirs(PASTA, exist_ok=True)
    feitos = []
    ser = serial.Serial(PORTA, BAUD, timeout=0.1)
    try:
        print("Abrindo serial... (o Arduino reinicia)")
        time.sleep(2.5)
        ser.reset_input_buffer()
        primeiro = True
        for nome, kp, ki, kd, beta, desc in ENSAIOS_CAL:
            caminho = os.path.join(PASTA, nome + ".csv")
            if os.path.exists(caminho):
                print(f"\n{nome}.csv ja existe, pulando")
                continue
            if not primeiro:
                descansa(ser, T_DESCANSO)
            primeiro = False
            if PAUSA:
                input(f"\nEnter pra rodar {nome} ({desc})")
            print(f"\n===== {nome}: P{kp} I{ki} D{kd} B{beta} =====")
            dados = []
            try:
                dados = roda_um(ser, kp, ki, kd, beta)
            except KeyboardInterrupt:
                print("\nInterrompido -- nao salvei esse ensaio.")
                break
            salva(dados, caminho)
            print(f"salvo: {caminho} ({len(dados)} amostras, {len(dados) * 0.015:.0f} s)")
            resumo(dados)
            feitos.append((nome, kp, ki, kd, beta, desc, np.array(dados)))
    finally:
        enviar(ser, f"M{ANG_DESCANSO}")   # termina com o servo parado
        ser.close()

    if not feitos:
        return
    print("\nlinhas pro ENSAIOS do ensaios.py:")
    for nome, kp, ki, kd, beta, desc, _ in feitos:
        print(f'    "{nome}": ("{nome}.csv", 2, [{kp}, {ki}, {kd}, 79, 30, 71.5, {beta}], "{desc}"),')

    fig, ax = plt.subplots(len(feitos), 1, figsize=(11, 2.2 * len(feitos)), squeeze=False)
    for a, (nome, kp, ki, kd, beta, desc, d) in zip(ax[:, 0], feitos):
        t = np.arange(len(d)) * 0.015
        a.plot(t, d[:, 2], "k--", lw=1)
        a.plot(t, d[:, 0], lw=1)
        a.set_ylabel("mm"); a.grid(alpha=.3)
        a.set_title(f"{nome}: {desc}", fontsize=9)
    ax[-1, 0].set_xlabel("tempo (s)")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()