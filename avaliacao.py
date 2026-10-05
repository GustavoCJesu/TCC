import numpy as np


def score_nasa(previsto, real):
    """Score oficial do desafio PHM08/C-MAPSS. É assimétrico de propósito:
    prever MAIS vida do que o motor tem (atraso, d > 0) é punido com exp(d/10),
    enquanto prever menos (adiantado) só custa exp(-d/13). Na prática uma
    manutenção adiantada é cara; uma atrasada é uma falha em voo.
    Menor é melhor."""
    d = previsto - real
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def avalia(previsto, real, teto, nome='Modelo'):
    """Calcula as métricas do conjunto de teste.

    IMPORTANTE: o alvo real também é limitado pelo teto. O modelo é treinado
    com RUL.clip(upper=teto) e portanto é INCAPAZ, por construção, de prever
    acima do teto. Compará-lo com o RUL real sem limite mede a métrica, não o
    modelo — e faz com que rodadas com tetos diferentes não sejam comparáveis
    entre si (ex.: com teto=50 o MAE trava em ~36 por causa disso).
    """
    real_lim = np.clip(real, 0, teto)
    erro = previsto - real_lim

    rmse = float(np.sqrt(np.mean(erro ** 2)))
    mae = float(np.mean(np.abs(erro)))
    score = score_nasa(previsto, real_lim)

    print(f'\n===== {nome} (teto = {teto}) =====')
    print(f'RMSE : {rmse:.2f} ciclos')
    print(f'MAE  : {mae:.2f} ciclos')
    print(f'Score: {score:.1f}  (assimétrico NASA, menor é melhor)')
    print(f'Viés : {erro.mean():+.2f} ciclos  (negativo = previsão conservadora)')

    n_acima = int((real > teto).sum())
    if n_acima:
        print(f'Obs. : {n_acima} motores tinham RUL real > {teto} e foram '
              f'limitados ao teto na avaliação.')

    faixas = calcula_faixas(previsto, real_lim)
    print('\nErro por faixa de RUL real:')
    print('  (a faixa baixa é a que importa para decidir manutenção;')
    print('   a faixa alta é dominada por incerteza irredutível)')
    for (lo, hi), v in faixas.items():
        print(f'  RUL {lo:3d}-{hi:<3d}  n={v["n"]:3d}   '
              f'RMSE={v["rmse"]:6.2f}   MAE={v["mae"]:6.2f}')

    return dict(rmse=rmse, mae=mae, score=score, faixas=faixas)


def calcula_faixas(previsto, real, limites=(0, 50, 100, 150)):
    """Quebra o erro por faixa de RUL. Sem isso o RMSE global esconde o fato
    de o erro ser ~4x maior na fase saudável do motor."""
    saida = {}
    for lo, hi in zip(limites[:-1], limites[1:]):
        m = (real >= lo) & (real < hi) if hi != limites[-1] else (real >= lo)
        if m.sum():
            e = previsto[m] - real[m]
            saida[(lo, hi)] = dict(n=int(m.sum()),
                                   rmse=float(np.sqrt(np.mean(e ** 2))),
                                   mae=float(np.mean(np.abs(e))))
    return saida


# ---------------------------------------------------------------------------
# Estado de saúde (classificação) e intervalo de previsão (quantis)
# ---------------------------------------------------------------------------

LIMITES_SAUDE = (50, 100)
SAUDAVEL, ATENCAO, PERIGOSO = 0, 1, 2
NOMES_SAUDE = ('Saudável (>=100)', 'Atenção (50-100)', 'Perigoso (<50)')


def classe_saude(rul, limites=LIMITES_SAUDE):
    return len(limites) - np.digitize(rul, limites)


def avalia_triagem(prob_nao_saudavel, rul_real, teto, limiar,
                   limite=LIMITES_SAUDE[-1]):
    real = np.clip(rul_real, 0, teto) < limite
    previsto = prob_nao_saudavel >= limiar

    vp = int((real & previsto).sum())
    fn = int((real & ~previsto).sum())
    fp = int((~real & previsto).sum())
    vn = int((~real & ~previsto).sum())
    acuracia = (vp + vn) / len(real)
    recall = vp / (vp + fn) if vp + fn else 0.0
    precisao = vp / (vp + fp) if vp + fp else 0.0

    print(f'\n===== MODELO 1: TRIAGEM (saudável x não saudável, '
          f'limiar={limiar:.2f}) =====')
    print(f'Acurácia : {acuracia:.1%}')
    print(f'Recall   : {recall:.1%}  (motores degradados que foram enviados ao modelo 2)')
    print(f'Precisão : {precisao:.1%}  (motores enviados que de fato estavam degradados)')
    print(f'Falsos negativos (degradado liberado como saudável): {fn}')
    print(f'Falsos positivos (saudável enviado ao modelo 2)    : {fp}')

    return dict(acuracia=acuracia, recall=recall, precisao=precisao,
                matriz=np.array([[vn, fp], [fn, vp]]))


def avalia_saude(classe_prevista, rul_real, teto, limites=LIMITES_SAUDE):
    real = classe_saude(np.clip(rul_real, 0, teto), limites)
    n = len(NOMES_SAUDE)
    matriz = np.array([[int(((real == i) & (classe_prevista == j)).sum())
                        for j in range(n)] for i in range(n)])
    acuracia = float((classe_prevista == real).mean())

    print('\n===== ESTADO DE SAÚDE (classificação) =====')
    print(f'Acurácia global: {acuracia:.1%}\n')
    cabecalho = 'real / previsto'
    print(f'{cabecalho:<20}' + ''.join(f'{s.split()[0]:>10}' for s in NOMES_SAUDE) + f'{"recall":>10}')
    for i, nome in enumerate(NOMES_SAUDE):
        total = matriz[i].sum()
        recall = matriz[i, i] / total if total else 0.0
        print(f'{nome:<20}' + ''.join(f'{v:>10}' for v in matriz[i]) + f'{recall:>9.1%}')

    confusao_grave = int(matriz[0, 2] + matriz[2, 0])
    print(f'\nConfusões perigoso <-> saudável: {confusao_grave} '
          f'(são as únicas que importam de verdade)')

    return dict(acuracia=acuracia, matriz=matriz,
                recall=[float(matriz[i, i] / matriz[i].sum()) if matriz[i].sum() else 0.0
                        for i in range(n)])


def avalia_intervalo(inferior, mediana, superior, rul_real, teto):
    """Avalia o intervalo de previsão de 80% (quantis 10% e 90%).

    A largura do intervalo é a forma do modelo dizer o quanto ele sabe:
    estreita perto da falha, larga quando o motor está saudável.
    """
    real = np.clip(rul_real, 0, teto)
    dentro = (real >= inferior) & (real <= superior)

    print('\n===== INTERVALO DE PREVISÃO (80%) =====')
    print(f'Cobertura real : {dentro.mean():.1%}  (esperado ~80%)')
    print(f'Largura média  : {(superior - inferior).mean():.1f} ciclos')
    print('\nPor faixa de RUL real:')
    faixas = {}
    for i, (lo, hi) in enumerate(zip((0,) + LIMITES_SAUDE, LIMITES_SAUDE + (10 ** 6,))):
        m = (real >= lo) & (real < hi)
        nome = NOMES_SAUDE[len(LIMITES_SAUDE) - i]
        if m.sum():
            faixas[nome] = dict(
                n=int(m.sum()),
                largura=float((superior - inferior)[m].mean()),
                cobertura=float(dentro[m].mean()))
            print(f'  {nome:<20} n={m.sum():3d}  '
                  f'largura={faixas[nome]["largura"]:6.1f} ciclos  '
                  f'cobertura={faixas[nome]["cobertura"]:.0%}')

    return dict(cobertura=float(dentro.mean()),
                largura=float((superior - inferior).mean()), faixas=faixas)
