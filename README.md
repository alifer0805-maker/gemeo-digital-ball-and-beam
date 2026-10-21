# Gêmeo digital de um sistema bola-viga

Código do Trabalho de Conclusão de Curso *Desenvolvimento e validação de estratégias de controle baseadas em gêmeo digital com ferramentas open-source: estudo de caso em um sistema bola-viga* (Engenharia Elétrica, UFPR, 2026).

Autor: Alifer Eduardo Coletta de Almeida
Orientador: Prof. Dr. Ricardo Schumacher

O gêmeo é um modelo de caixa cinza da bancada (bola com atrito de Karnopp, servo com atraso e velocidade limitada, sensor ultrassônico com ruído e o firmware do Arduino reproduzido linha a linha). Os parâmetros físicos são calibrados por evolução diferencial a partir de ensaios em malha fechada, e o gêmeo calibrado é usado para escolher os ganhos do PID, de forma fixa ou a cada mudança de referência.

## Instalação

```
pip install -r requirements.txt
```

Os scripts foram usados no Spyder, rodando cada arquivo com F5 a partir desta pasta. Na primeira execução o Numba compila o modelo (10 a 20 s). Depois disso, um ensaio de 12 s é simulado em cerca de 0,2 ms.

## Arquivos

| Arquivo | Função |
|---|---|
| `modelo.py` | O gêmeo: planta, servo, sensor e o PID de cada versão do firmware |
| `metricas.py` | Métricas por degrau (Mp, ts, tr, ef, IAE, batida do servo) e as funções custo J e J2 |
| `ensaios.py` | Catálogo dos ensaios: arquivo, versão do firmware e ganhos de cada um |
| `1_calibracao.py` | Calibração dos 10 parâmetros físicos por evolução diferencial |
| `2_otimizacao.py` | Otimização dos ganhos no gêmeo e teste de robustez |
| `3_avaliar_ensaio.py` | Compara um ensaio da bancada com o gêmeo |
| `4_rodadas_e_figuras.py` | Tabela das rodadas de validação e figuras |
| `ganhos_por_setpoint.py` | Otimização rápida dos ganhos para um único degrau |
| `5_ensaio_adaptativo.py` | Ensaio na bancada com os ganhos escolhidos pelo gêmeo a cada referência |
| `visual.py` | Janela ao vivo com a bola real e a prevista pelo gêmeo |
| `ensaio_ballbeam.py` | Ensaio automático na bancada com ganhos fixos |
| `ensaios_calibracao.py` | Sequência de ensaios para calibração, com descanso do servo entre eles |
| `firmware/ballbeam_v8/` | Firmware v8 do Arduino Uno |

## Pastas

- `dados/`: arquivos CSV dos ensaios (posição, ângulo do servo, referência e duração do ciclo)
- `parametros/`: parâmetros calibrados em cada rodada (`fis_rodada1.npy` ...)
- `resultados/`: figuras e ganhos gerados pelos scripts

## Ciclo de uso

1. Calibrar o gêmeo com os ensaios disponíveis (`1_calibracao.py`).
2. Otimizar os ganhos no gêmeo (`2_otimizacao.py`).
3. Ensaiar os ganhos escolhidos na bancada (`ensaio_ballbeam.py`).
4. Comparar bancada e gêmeo (`3_avaliar_ensaio.py`). Se divergirem, o ensaio novo entra na próxima calibração.

## Comandos do firmware

| Comando | Exemplo | Efeito |
|---|---|---|
| `P` | `P0.26` | define Kp |
| `I` | `I0.01` | define Ki |
| `D` | `D0.35` | define Kd |
| `B` | `B0.95` | define β do filtro da derivada |
| `S` | `S200` | define a referência (mm) e zera a integral |
| `N` | `N2.5` | soma um ajuste, em graus, ao ângulo de nível |
| `M` | `M75` | modo manual, com o servo fixo no ângulo dado |
| `A` | `A` | volta ao automático |

Comunicação serial a 2 000 000 baud. O firmware lê um comando por ciclo, então os scripts esperam 0,15 s entre comandos.

## Licença

MIT
