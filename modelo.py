# -*- coding: utf-8 -*-
"""
GEMEO DIGITAL DO BALL-AND-BEAM -- o modelo (caixa cinza).

Cada peca espelha uma parte da bancada:

  PLANTA    a = K*sin(theta0(p) - theta)  [mm/s^2]
            com atrito de Karnopp (estatico a_s, cinetico a_c, viscoso b) e
            batentes nas pontas da viga. theta0(p) = c0 + c1*(p - 165) e' o
            angulo de equilibrio, que muda ao longo da viga.
  SERVO     comando inteiro (como o servo.write), atraso puro de D amostras e
            velocidade maxima omega [graus/s].
  SENSOR    leitura = posicao real + ruido gaussiano (desvio s0, mais kv com a
            bola andando), quantizada em passos de Q mm, e depois EXATAMENTE o
            aplicarFiltro() do firmware (faixa valida, rejeicao de saltos com
            3 confirmacoes, trava com a bola colada, media exponencial 0.55).
  FIRMWARE  o PID de cada versao reproduzido linha a linha, Ts = 15 ms:
            variante 0 = codigo anterior ("Ts fixo")
            variante 1 = v7 (nivel por setpoint, derivada em janela, banda da integral)
            variante 2 = v8 (v7 + passa-baixa na derivada)

Vetor de parametros fisicos (o que a calibracao estima):
  fis = [K, a_s, a_c, b, c0, c1, D, kv, pdrop, omega]
  (pdrop = probabilidade de perder uma leitura; ficou 0 nas calibracoes)
"""
import numpy as np
from numba import njit, prange

TS = 0.015                        # periodo de amostragem do firmware [s]
P_MIN, P_MAX = 25.0, 316.0        # batentes da viga [mm]
OMEGA = 300.0                     # velocidade do servo se fis nao trouxer [graus/s]
# filtro do firmware
ALPHA = 0.55
DIST_MIN, DIST_MAX = 35.0, 320.0
SALTO, NCONF, ZCOL = 40.0, 3, 30.0

NOMES_FIS = ["K", "a_s", "a_c", "b", "c0", "c1", "D", "kv", "pdrop", "omega"]


@njit(cache=True)
def _filtro(raw, st):
    """aplicarFiltro() do firmware. st = [distFiltrada, primeira, ultimoCandidato, candidatos]"""
    if raw < DIST_MIN or raw > DIST_MAX:
        return st[0]
    if st[1] > 0.5:
        st[0] = raw; st[2] = raw; st[1] = 0.0
        return st[0]
    if abs(raw - st[0]) <= SALTO:
        st[0] = ALPHA * raw + (1 - ALPHA) * st[0]; st[3] = 0.0
    else:
        if st[0] < DIST_MIN + ZCOL:
            st[3] = 0.0
            return st[0]
        if abs(raw - st[2]) <= SALTO:
            st[3] += 1.0
        else:
            st[3] = 1.0
        st[2] = raw
        if st[3] >= NCONF:
            st[0] = raw; st[3] = 0.0
    return st[0]


@njit(cache=True)
def _planta(p, v, th, h, K, a_s, a_c, b, c0, c1):
    """Um passo de integracao (Euler, passo h) da bola na viga."""
    th0 = c0 + c1 * (p - 165.0)
    a_d = K * np.sin((th0 - th) * np.pi / 180.0)      # aceleracao "motriz"
    if v == 0.0:                                       # parada: vence o atrito estatico?
        if abs(a_d) <= a_s:
            a = 0.0
        else:
            a = a_d - a_c * np.sign(a_d)
    else:                                              # andando: atrito cinetico + viscoso
        a = a_d - a_c * np.sign(v) - b * v
    vn = v + a * h
    if v != 0.0 and v * vn < 0.0:                      # atrito nao inverte o movimento
        vn = 0.0
    p = p + vn * h
    if p < P_MIN:
        p = P_MIN
        if vn < 0: vn = 0.0
    elif p > P_MAX:
        p = P_MAX
        if vn > 0: vn = 0.0
    return p, vn


@njit(cache=True)
def simula(fis, ctrl, variante, sp, p0, ang0, seed, s0, Q):
    """Simula a MALHA FECHADA (planta + servo + sensor + firmware).

    fis      parametros fisicos (ver topo do arquivo)
    ctrl     [Kp, Ki, Kd, N1, LIM, N2, beta]
               variante 0: N1 = NIVEL fixo (N2 e beta ignorados)
               variante 1/2: N1 = NIVEL_PERTO (alvo 80), N2 = NIVEL_LONGE (alvo 250)
               LIM = limite do servo em torno do nivel; beta = filtro da derivada (v8)
    variante 0, 1 ou 2 (ver topo)
    sp       setpoint a cada amostra [mm]
    p0, ang0 posicao e angulo iniciais
    seed     semente do ruido (mude para outra realizacao)
    s0       desvio-padrao do ruido do sensor com a bola parada [mm]
    Q        passo de quantizacao da leitura [mm]
    Retorna (y = posicao que o firmware "ve", u = angulo comandado, p = posicao real).
    """
    np.random.seed(seed)
    K, a_s, a_c, b, c0, c1 = fis[0], fis[1], fis[2], fis[3], fis[4], fis[5]
    D = int(fis[6])
    kv = fis[7]
    pdrop = fis[8]
    om = fis[9]
    Kp, Ki, Kd, N1, LIM, N2 = ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4], ctrl[5]
    beta = ctrl[6] if ctrl.shape[0] > 6 else 0.0
    n = sp.shape[0]
    y = np.empty(n); u = np.empty(n); pr = np.empty(n)
    p = p0; v = 0.0; th = ang0
    fila = np.full(D + 1, np.round(ang0))          # atraso do servo
    st = np.array([0.0, 1.0, 0.0, 0.0])            # estado do filtro do sensor
    integ = 0.0; eant = 0.0; sat = 0; derf = 0.0
    hp = np.full(6, p0); ih = 0                    # historico da derivada em janela
    spant = sp[0]
    nsub = 3; h = TS / nsub                         # 3 sub-passos por amostra
    for k in range(n):
        # servo + planta durante um periodo
        cmd = fila[0]
        for _ in range(nsub):
            dmax = om * h
            th += max(-dmax, min(dmax, cmd - th))
            p, v = _planta(p, v, th, h, K, a_s, a_c, b, c0, c1)
        # sensor
        raw = p + (s0 + kv * min(abs(v) / 10.0, 1.0)) * np.random.randn()
        if Q > 0: raw = Q * np.round(raw / Q)
        if np.random.rand() < pdrop: raw = -1.0     # leitura perdida (sem eco)
        yk = _filtro(raw, st)
        # firmware
        e = sp[k] - yk
        if variante == 0:                            # codigo anterior
            if not ((sat > 0 and e > 0) or (sat < 0 and e < 0)):
                integ += e * TS
            der = (e - eant) / TS; eant = e
            NV = N1
        else:                                        # v7 / v8
            if sp[k] != spant:
                integ = 0.0
            spant = sp[k]
            if abs(e) >= 60.0:
                integ = 0.0
            elif not ((sat > 0 and e > 0) or (sat < 0 and e < 0)):
                integ += e * TS
            der_cru = -(yk - hp[ih]) / (6 * TS)
            hp[ih] = yk; ih = (ih + 1) % 6
            if variante == 2:
                derf = beta * derf + (1.0 - beta) * der_cru
                der = derf
            else:
                der = der_cru
            NV = N1 + (N2 - N1) * (sp[k] - 80.0) / 170.0
        ang = NV - (Kp * e + Ki * integ + Kd * der)  # SENTIDO = -1
        if ang > NV + LIM:
            ang = NV + LIM; sat = 1
        elif ang < NV - LIM:
            ang = NV - LIM; sat = -1
        else:
            sat = 0
        for j in range(D):
            fila[j] = fila[j + 1]
        fila[D] = np.round(ang)
        y[k] = yk; u[k] = ang; pr[k] = p
    return y, u, pr


@njit(parallel=True, cache=True)
def simula_lote(fis, ctrls, variantes, sp_tudo, ini, fim, p0s, a0s, sementes, s0, Q):
    """Roda simula() para todos os ensaios x sementes em paralelo (um por nucleo).
    Os setpoints de todos os ensaios vem emendados em sp_tudo; o ensaio e vai de
    ini[e] a fim[e]. Devolve Y e U com uma linha por semente."""
    ne = ini.shape[0]
    ns = sementes.shape[0]
    Y = np.empty((ns, sp_tudo.shape[0]))
    U = np.empty((ns, sp_tudo.shape[0]))
    for j in prange(ne * ns):
        e = j // ns
        k = j % ns
        y, u, _ = simula(fis, ctrls[e], variantes[e], sp_tudo[ini[e]:fim[e]],
                         p0s[e], a0s[e], sementes[k], s0, Q)
        Y[k, ini[e]:fim[e]] = y
        U[k, ini[e]:fim[e]] = u
    return Y, U


def carrega_fis(caminho):
    """Le um vetor salvo pela calibracao. Retorna (fis para o simula, s0)."""
    x = np.load(caminho)
    K, a_s, a_c, b, c0, c1, D, kv, s0, om = x
    return np.array([K, a_s, a_c, b, c0, c1, round(D), kv, 0.0, om]), float(s0)
