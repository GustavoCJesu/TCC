from dados import coletaDados
from normalização import normalizacao
from modelo import modelo
from graficos import geraGraficos

import numpy as np
import pandas as pd

janela = int(input('Tamanho de janela: '))
teto = int(input('Teto Maximo do RUL: '))



df, df_teste, rul_verdadeiro = coletaDados(teto)

print()

x_treino, x_val, y_treino_norm, y_val_norm, scaler, col_sensores, col_remove, y_val, x_teste = normalizacao(df, df_teste, janela, teto)

print('========================================================')
print(y_val.min(), y_val.max(), np.unique(y_val)[:20])

historico, rul_verdadeiro, prev_teste = modelo(janela, teto, x_treino, x_val, y_treino_norm, y_val_norm, y_val, df, x_teste, rul_verdadeiro)

geraGraficos(historico, rul_verdadeiro, prev_teste)

