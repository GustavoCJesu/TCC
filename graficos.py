import os

import matplotlib

# TkAgg abre janela; se não houver display (servidor, terminal remoto) cai
# para Agg e apenas salva os arquivos, em vez de quebrar.
_INTERATIVO = bool(os.environ.get('DISPLAY')) and os.environ.get('MPLBACKEND') != 'Agg'
matplotlib.use('TkAgg' if _INTERATIVO else 'Agg')

import matplotlib.pyplot as plt
import numpy as np

# Paleta categórica validada para daltonismo (ΔE 24.7 protan, 33.6 visão normal)
AZUL = '#2a78d6'
LARANJA = '#eb6834'
# Rampa sequencial de um único tom, claro -> escuro, para grandezas ordenadas
RAMPA = ['#86b6ef', '#3987e5', '#184f95']
CINZA = '#52514e'


def _finaliza(nome_arquivo):
    plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight')
    if _INTERATIVO:
        plt.show()
    plt.close()
    print(f'  gráfico salvo: {nome_arquivo}')


def geraGraficos(historicos, rul_verdadeiro, previsoes, janela, teto,
                 metricas=None):
    """historicos: lista com o histórico de cada modelo do ensemble.
    previsoes: dict com 'classe', 'inferior', 'mediana', 'superior'."""

    prev_teste = previsoes['mediana']

    sufixo = f'Teto-{teto}-Janela-{janela}'

    # O modelo não pode prever acima do teto, então a referência também é
    # limitada — mesmo critério usado nas métricas.
    real = np.clip(rul_verdadeiro, 0, teto)

    print('\n===== GRÁFICOS =====')

    # ---------- 1. Curva de aprendizado (média do ensemble) ----------
    n_ep = min(len(h.history['loss']) for h in historicos)
    treino = np.array([h.history['loss'][:n_ep] for h in historicos])
    val = np.array([h.history['val_loss'][:n_ep] for h in historicos])
    ep = np.arange(1, n_ep + 1)

    plt.figure(figsize=(10, 5))
    for serie, cor, rotulo in [(treino, AZUL, 'Treino'),
                               (val, LARANJA, 'Validação')]:
        plt.fill_between(ep, serie.min(axis=0), serie.max(axis=0),
                         color=cor, alpha=0.18, linewidth=0)
        plt.plot(ep, serie.mean(axis=0), color=cor, linewidth=2, label=rotulo)

    plt.title(f'Curva de Aprendizado — média de {len(historicos)} modelos '
              f'(faixa = mínimo a máximo)')
    plt.xlabel('Época')
    plt.ylabel('Perda (MSE, escala normalizada)')
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.25, linewidth=0.6)
    plt.gca().set_axisbelow(True)
    _finaliza(f'grafico_curva_treino-{sufixo}.png')

    # ---------- 2. Previsto vs Real ----------
    plt.figure(figsize=(7, 7))
    lim = max(real.max(), prev_teste.max()) * 1.03
    plt.plot([0, lim], [0, lim], '--', color=CINZA, linewidth=1.5,
             label='Previsão perfeita', zorder=1)
    plt.scatter(real, prev_teste, s=45, color=AZUL, alpha=0.75,
                edgecolors='white', linewidths=0.8, zorder=2,
                label='Motor de teste')
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.title('RUL Previsto vs. RUL Real (Teste)')
    plt.xlabel('RUL Real (ciclos)')
    plt.ylabel('RUL Previsto (ciclos)')
    plt.legend(frameon=False, loc='upper left')
    plt.grid(True, alpha=0.25, linewidth=0.6)
    plt.gca().set_axisbelow(True)
    _finaliza(f'grafico_previsto_vs_real-{sufixo}.png')

    # ---------- 3. Previsões ordenadas ----------
    ordem = np.argsort(real)

    plt.figure(figsize=(12, 5))
    plt.fill_between(np.arange(len(ordem)),
                     previsoes['inferior'][ordem], previsoes['superior'][ordem],
                     color=LARANJA, alpha=0.18, linewidth=0,
                     label='Intervalo de 80%')
    plt.plot(real[ordem], color=AZUL, linewidth=2, label='RUL Real')
    plt.plot(prev_teste[ordem], color=LARANJA, linewidth=1.5, alpha=0.9,
             label='RUL Previsto (mediana)')
    plt.title('RUL Previsto vs. Real com incerteza — 100 motores (ordenados)')
    plt.xlabel('Motores (ordenados por RUL real crescente)')
    plt.ylabel('RUL (ciclos)')
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.25, linewidth=0.6)
    plt.gca().set_axisbelow(True)
    _finaliza(f'grafico_previsoes_ordenadas-{sufixo}.png')

    # ---------- 4. Erro por faixa de RUL ----------
    # O RMSE global esconde que o erro é várias vezes maior na fase saudável
    # do motor. Este é o gráfico que sustenta essa conclusão.
    if metricas and metricas.get('faixas'):
        faixas = metricas['faixas']
        rotulos = [f'{lo}–{hi}' for lo, hi in faixas]
        valores = [v['rmse'] for v in faixas.values()]
        contagens = [v['n'] for v in faixas.values()]
        cores = RAMPA[:len(valores)]

        plt.figure(figsize=(8, 5))
        barras = plt.bar(rotulos, valores, color=cores, width=0.6)
        for barra, valor, n in zip(barras, valores, contagens):
            plt.text(barra.get_x() + barra.get_width() / 2,
                     valor + max(valores) * 0.02,
                     f'{valor:.1f}\n(n={n})', ha='center', va='bottom',
                     fontsize=10, color=CINZA)

        plt.ylim(0, max(valores) * 1.25)
        plt.title('Erro por faixa de RUL — o modelo é preciso perto da falha')
        plt.xlabel('Faixa de RUL real (ciclos)')
        plt.ylabel('RMSE (ciclos)')
        plt.grid(True, axis='y', alpha=0.25, linewidth=0.6)
        plt.gca().set_axisbelow(True)
        _finaliza(f'grafico_erro_por_faixa-{sufixo}.png')

    # ---------- 5. Matriz de confusão do estado de saúde ----------
    # O que interessa aqui são os CANTOS: crítico previsto como saudável
    # (falha não detectada) e saudável previsto como crítico (alarme falso).
    if metricas and metricas.get('saude'):
        from matplotlib.colors import LinearSegmentedColormap
        from avaliacao import NOMES_SAUDE

        matriz = np.asarray(metricas['saude']['matriz'])
        mapa = LinearSegmentedColormap.from_list(
            'azul_seq', ['#ffffff', '#cde2fb', '#3987e5', '#0d366b'])

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(matriz, cmap=mapa, vmin=0, vmax=matriz.max())

        curtos = [n.split()[0] for n in NOMES_SAUDE]
        ax.set_xticks(range(len(curtos)), curtos)
        ax.set_yticks(range(len(curtos)), curtos)
        ax.set_xlabel('Estado previsto')
        ax.set_ylabel('Estado real')
        ax.set_title(f'Matriz de Confusão — acurácia '
                     f'{metricas["saude"]["acuracia"]:.0%}')

        limiar = matriz.max() * 0.55
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                ax.text(j, i, str(matriz[i, j]), ha='center', va='center',
                        fontsize=15,
                        color='white' if matriz[i, j] > limiar else '#0b0b0b')

        ax.set_xticks(np.arange(-.5, len(curtos), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(curtos), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=2)
        ax.tick_params(which='minor', length=0)
        _finaliza(f'grafico_matriz_confusao-{sufixo}.png')
