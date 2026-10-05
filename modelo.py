from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import tensorflow as tf

import numpy as np

from avaliacao import (avalia, avalia_saude, avalia_triagem, classe_saude,
                       NOMES_SAUDE, LIMITES_SAUDE, SAUDAVEL, ATENCAO, PERIGOSO)

LIMITE_SAUDE = LIMITES_SAUDE[-1]
LIMITE_PERIGO = LIMITES_SAUDE[0]
LIMIAR_TRIAGEM = 0.5
LIMIARES_CANDIDATOS = [round(0.2 + 0.05 * i, 2) for i in range(13)]


def camadas_lstm(empilhada):
    if empilhada:
        return [LSTM(64, return_sequences=True), LSTM(32)]
    return [LSTM(64)]


def constroi_triagem(janela, n_features, semente, empilhada=False):
    tf.keras.utils.set_random_seed(semente)
    rede = Sequential([
        Input((janela, n_features)),
        *camadas_lstm(empilhada),
        Dense(1, activation='sigmoid'),
    ])
    rede.compile(optimizer='Adam', loss='binary_crossentropy',
                 metrics=['accuracy'])
    return rede


def constroi_diagnostico(janela, n_features, semente, empilhada=False):
    tf.keras.utils.set_random_seed(semente)
    rede = Sequential([
        Input((janela, n_features)),
        *camadas_lstm(empilhada),
        Dense(1),
    ])
    rede.compile(optimizer='Adam', loss='mse', metrics=['mae'])
    return rede


def treina_ensemble(construtor, n_modelos, semente, x_treino, y_treino,
                    x_val, y_val, epocas, verbose, pesos_classe=None):
    redes, historicos = [], []
    for i in range(n_modelos):
        rede = construtor(semente + i)
        if verbose and i == 0:
            rede.summary()
        historicos.append(rede.fit(
            x_treino, y_treino, validation_data=(x_val, y_val),
            epochs=epocas, batch_size=256, verbose=verbose,
            class_weight=pesos_classe))
        redes.append(rede)
    return redes, historicos


def previsoes_brutas(x, redes_1, redes_2, teto):
    prob = np.mean([r.predict(x, verbose=0) for r in redes_1], axis=0).ravel()
    rul = np.mean([r.predict(x, verbose=0) for r in redes_2], axis=0).ravel() * teto
    return prob, rul


def decide(prob, rul, limiar, devolve=False):
    enviado = prob >= limiar
    classe = np.full(len(prob), SAUDAVEL)
    classe[enviado] = np.where(rul[enviado] < LIMITE_PERIGO, PERIGOSO, ATENCAO)
    if devolve:
        classe[enviado & (rul >= LIMITE_SAUDE)] = SAUDAVEL
    return enviado, classe, np.where(enviado, rul, np.nan)


def recall_medio(classe, real):
    return float(np.mean([(classe[real == k] == k).mean()
                          for k in range(len(NOMES_SAUDE)) if (real == k).any()]))


def escolhe_limiar(prob, rul, rul_real, teto, devolve=False):
    real = classe_saude(np.clip(rul_real, 0, teto))
    candidatos = LIMIARES_CANDIDATOS + ([0.1, 0.15] if devolve else [])
    candidatos = sorted(candidatos, key=lambda l: abs(l - 0.5))
    notas = [recall_medio(decide(prob, rul, l, devolve)[1], real)
             for l in candidatos]
    return candidatos[int(np.argmax(notas))]


def modelo(janela, teto, x_treino, x_val, y_treino_norm, y_val_norm, y_val,
           df, x_teste, rul_verdadeiro, epocas=30,
           limiar_triagem=LIMIAR_TRIAGEM, limiar_auto=False, semente=42,
           n_modelos=1, empilhada=False, margem=0, devolve=False,
           peso_degradado=1.0, verbose=1):

    if teto < LIMITE_SAUDE:
        raise ValueError(
            f'teto={teto} é menor que o limite de motor saudável '
            f'({LIMITE_SAUDE}): nenhum motor seria saudável e o modelo 1 '
            f'não teria o que aprender. Use teto >= {LIMITE_SAUDE}.')

    n_features = x_treino.shape[2]

    rul_treino = np.rint(y_treino_norm * teto)
    rul_val = np.rint(y_val_norm * teto)

    print('\n' + '=' * 60)
    print('MODELO 1 — triagem (saudável x não saudável)')
    print('=' * 60)

    redes_1, hist_1 = treina_ensemble(
        lambda s: constroi_triagem(janela, n_features, s, empilhada),
        n_modelos, semente,
        x_treino, (rul_treino < LIMITE_SAUDE).astype('float32'),
        x_val, (rul_val < LIMITE_SAUDE).astype('float32'), epocas, verbose,
        pesos_classe={0: 1.0, 1: peso_degradado})

    m_tr = rul_treino < LIMITE_SAUDE + margem
    m_val = rul_val < LIMITE_SAUDE + margem
    print('\n' + '=' * 60)
    print(f'MODELO 2 — diagnóstico (RUL) | {m_tr.sum()} janelas de treino, '
          f'{m_val.sum()} de validação')
    print('=' * 60)

    redes_2, hist_2 = treina_ensemble(
        lambda s: constroi_diagnostico(janela, n_features, s, empilhada),
        n_modelos, semente,
        x_treino[m_tr], y_treino_norm[m_tr],
        x_val[m_val], y_val_norm[m_val], epocas, verbose)

    prob_val, rul_val_prev = previsoes_brutas(x_val, redes_1, redes_2, teto)
    limiar = limiar_triagem
    if limiar_auto:
        limiar = escolhe_limiar(prob_val, rul_val_prev, rul_val, teto, devolve)
        print(f'\nLimiar de triagem escolhido na validação: {limiar:.2f}')

    _, classe_val, _ = decide(prob_val, rul_val_prev, limiar, devolve)
    real_val = classe_saude(rul_val)
    validacao = dict(acuracia=float((classe_val == real_val).mean()),
                     recall_medio=recall_medio(classe_val, real_val))

    n = len(x_teste)
    prob_nao_saudavel, rul_teste = previsoes_brutas(x_teste, redes_1, redes_2, teto)
    enviado, classe_prevista, rul_previsto = decide(prob_nao_saudavel,
                                                    rul_teste, limiar, devolve)

    metricas = {'limiar': limiar, 'validacao': validacao}
    metricas['triagem'] = avalia_triagem(prob_nao_saudavel, rul_verdadeiro,
                                         teto, limiar)
    metricas['saude'] = avalia_saude(classe_prevista, rul_verdadeiro, teto)

    print(f'\n{enviado.sum()} de {n} motores foram enviados ao modelo 2.')
    if enviado.any():
        metricas.update(avalia(
            rul_previsto[enviado], rul_verdadeiro[enviado], teto,
            nome=f'MODELO 2 — RUL dos {enviado.sum()} motores enviados'))

    print('\n===== LAUDO POR MOTOR (primeiros 10) =====')
    print(f'{"motor":>6} {"estado":<18} {"P(n/saud.)":>10} {"RUL previsto":>16} {"real":>6}')
    for i in range(min(10, n)):
        if enviado[i]:
            previsto = f'{rul_previsto[i]:.0f}'
        else:
            previsto = f'>= {LIMITE_SAUDE} (n/ est.)'
        print(f'{i + 1:>6} {NOMES_SAUDE[classe_prevista[i]]:<18} '
              f'{prob_nao_saudavel[i]:>10.0%} {previsto:>16} {rul_verdadeiro[i]:>6.0f}')

    previsoes = dict(classe=classe_prevista, enviado=enviado,
                     prob_nao_saudavel=prob_nao_saudavel, rul=rul_previsto)
    historicos = dict(triagem=hist_1, diagnostico=hist_2)

    return historicos, rul_verdadeiro, previsoes, metricas
