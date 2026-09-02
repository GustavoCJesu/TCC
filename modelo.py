from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import numpy as np
import pandas as pd

from avaliacao import (avalia, avalia_saude, avalia_intervalo,
                       classe_saude, NOMES_SAUDE)

# Quantis previstos pela cabeça de regressão: 10%, 50% (mediana) e 90%.
# O par 10/90 forma o intervalo de confiança de 80%.
QUANTIS = (0.1, 0.5, 0.9)

# Peso relativo da perda de regressão. A entropia cruzada opera na casa de
# 0.4 e a pinball na de 0.02; sem esse fator a cabeça de RUL seria ignorada.
PESO_RUL = 20.0


def perda_pinball(y_real, y_previsto):
    """Pinball loss (quantile loss). Para o quantil q, erros para baixo pesam
    q e para cima pesam (1-q). Treinando q=0.1 e q=0.9 em paralelo, a rede
    aprende os limites do intervalo em vez de um único valor médio."""
    q = tf.constant([QUANTIS], dtype=tf.float32)
    e = y_real - y_previsto
    return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))


def constroi_rede(janela, n_features, dropout, seed):
    """Rede de DUAS CABEÇAS sobre um tronco LSTM compartilhado:

      saude -> 3 classes (crítico / atenção / saudável)
      rul   -> 3 quantis (10%, 50%, 90%) do RUL normalizado

    O tronco é o mesmo porque as duas tarefas leem a mesma degradação; só a
    pergunta feita no fim é diferente.
    """
    tf.keras.utils.set_random_seed(seed)

    entrada = Input((janela, n_features))
    x = LSTM(64, return_sequences=True)(entrada)
    x = Dropout(dropout)(x)
    x = LSTM(32)(x)
    tronco = Dropout(dropout)(x)

    saida_saude = Dense(len(NOMES_SAUDE), activation='softmax', name='saude')(tronco)
    saida_rul = Dense(len(QUANTIS), name='rul')(tronco)

    rede = Model(entrada, [saida_saude, saida_rul])
    rede.compile(
        optimizer='Adam',
        loss={'saude': 'sparse_categorical_crossentropy', 'rul': perda_pinball},
        loss_weights={'saude': 1.0, 'rul': PESO_RUL},
        metrics={'saude': 'accuracy'}
    )
    return rede


def modelo(janela, teto, x_treino, x_val, y_treino_norm, y_val_norm, y_val,
           df, x_teste, rul_verdadeiro, n_modelos=5, dropout=0.4):

    # Alvos: a classe de saúde vem do RUL em ciclos; os quantis, do normalizado
    alvos_treino = {'saude': classe_saude(y_treino_norm * teto),
                    'rul': y_treino_norm.reshape(-1, 1)}
    alvos_val = {'saude': classe_saude(y_val_norm * teto),
                 'rul': y_val_norm.reshape(-1, 1)}

    historicos, probs, quantis = [], [], []

    for i in range(n_modelos):
        seed = 42 + i
        print(f'\n########## Treinando modelo {i + 1}/{n_modelos} '
              f'(seed={seed}) ##########\n')

        rede = constroi_rede(janela, x_treino.shape[2], dropout, seed)
        if i == 0:
            rede.summary()

        historico = rede.fit(
            x_treino, alvos_treino,
            validation_data=(x_val, alvos_val),
            epochs=100,
            batch_size=256,
            callbacks=[EarlyStopping('val_loss', patience=20,
                                     restore_best_weights=True)],
            verbose=1
        )

        historicos.append(historico)
        p_saude, p_rul = rede.predict(x_teste, verbose=0)
        probs.append(p_saude)
        quantis.append(p_rul)

    # Média do ensemble
    probs = np.mean(probs, axis=0)
    classe_prevista = probs.argmax(axis=1)

    # np.sort garante que os quantis não se cruzem (10% <= 50% <= 90%)
    q = np.sort(np.clip(np.mean(quantis, axis=0) * teto, 0, teto), axis=1)
    inferior, mediana, superior = q[:, 0], q[:, 1], q[:, 2]

    metricas = avalia(mediana, rul_verdadeiro, teto,
                      nome=f'RUL — mediana do ensemble de {n_modelos} modelos')
    metricas['saude'] = avalia_saude(classe_prevista, rul_verdadeiro, teto)
    metricas['intervalo'] = avalia_intervalo(inferior, mediana, superior,
                                             rul_verdadeiro, teto)

    print('\n===== LAUDO POR MOTOR (primeiros 10) =====')
    print(f'{"motor":>6} {"estado":<18} {"conf.":>6} {"RUL previsto":>22} {"real":>6}')
    for i in range(10):
        faixa = f'{inferior[i]:.0f}-{superior[i]:.0f} (med {mediana[i]:.0f})'
        print(f'{i + 1:>6} {NOMES_SAUDE[classe_prevista[i]]:<18} '
              f'{probs[i].max():>5.0%} {faixa:>22} {rul_verdadeiro[i]:>6.0f}')

    previsoes = dict(classe=classe_prevista, probabilidades=probs,
                     inferior=inferior, mediana=mediana, superior=superior)

    return historicos, rul_verdadeiro, previsoes, metricas
