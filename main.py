from dados import coletaDados
from normalização import normalizacao
from modelo import modelo
from graficos import geraGraficos

import numpy as np
import pandas as pd


def pergunta(texto, padrao):
    resposta = input(f'{texto} [{padrao}]: ').strip()
    return int(resposta) if resposta else padrao


janela = pergunta('Tamanho de janela', 25)
teto = pergunta('Teto Maximo do RUL', 150)
# Uma única LSTM varia ~±0.5 de RMSE só por causa da inicialização.
# A média de vários modelos remove essa loteria e melhora o resultado.
n_modelos = pergunta('Quantos modelos no ensemble', 5)

df, df_teste, rul_verdadeiro = coletaDados(teto)

x_treino, x_val, y_treino_norm, y_val_norm, scaler, col_sensores, col_remove, y_val, x_teste = normalizacao(
    df, df_teste, janela, teto)

print(f'\nJanelas de treino: {len(x_treino)} | validação: {len(x_val)} | '
      f'motores de teste: {len(x_teste)}')
print(f'Features por passo: {x_treino.shape[2]}')

historicos, rul_verdadeiro, previsoes, metricas = modelo(
    janela, teto, x_treino, x_val, y_treino_norm, y_val_norm, y_val,
    df, x_teste, rul_verdadeiro, n_modelos=n_modelos)

geraGraficos(historicos, rul_verdadeiro, previsoes, janela, teto, metricas)
