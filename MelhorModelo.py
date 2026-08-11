import pandas as pd
import numpy as np

col_nomes = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']

col_nomes += [f'sensor_{i}' for i in range(1, 22)]

df = pd.read_csv('train_FD001.txt',
                 sep=r'\s+',
                 header=None,
                 names=col_nomes
                 )

print("Formato (linhas, colunas): ", df.shape)
print("\nPrimeiras 5 linhas: ")
print(df.head())
print("\nNumeros de motores", df['unit_number'].unique())

ciclo_max = df.groupby('unit_number')['time_in_cycles'].max()

df['ciclo_max'] = df['unit_number'].map(ciclo_max)
df['RUL'] = df['ciclo_max'] - df['time_in_cycles']

df = df.drop(columns=['ciclo_max'])

print('\n Ultimas linhas do motor 1 (RUL deve chegar a 0): ')
print(df[df['unit_number'] == 1][['unit_number', 'time_in_cycles', 'RUL']].tail())

RUL_TETO = 125.0

df['RUL'] = df['RUL'].clip(upper=RUL_TETO)

print('Novo RUL maximo (Deve ser 125): ', df['RUL'].max())
print('RUL minimo (deve continuar 0): ', df['RUL'].min())

print('\n Primeiras linhas do motor 1 (RUL agora deve estar "travado" em 125): ')
print(df[df['unit_number'] == 1][['unit_number', 'time_in_cycles', 'RUL']].head())

col_sensores = [f'sensor_{i}' for i in range(1, 22)]
col_settings = ['op_setting_1', 'op_setting_2', 'op_setting_3']

print('Desvio padrão dos sensores (ordenado do menor para o maior): \n')
desvios = df[col_settings + col_sensores].std().sort_values()
print(desvios)

col_remove = [
    'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19', 'op_setting_1', 'op_setting_2', 'op_setting_3'
]

df = df.drop(columns=col_remove)
col_sensores_uteis = [c for c in df.columns if c.startswith('sensor_')]

print('Colunas restantes no DataFrame: ')
print(list(df.columns))
print('\n Numero de sensores úteis restantes: ', len(col_sensores_uteis))
print('sensores mantidos: ', col_sensores_uteis)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[col_sensores_uteis] = scaler.fit_transform(df[col_sensores_uteis])

print('Valores Minimos por sensor (devem ser ~0): ')
print(df[col_sensores_uteis].min().round(3))
print('\nValores maximos por sensor (devem ser ~1): ')
print('\nPrimeiras linhas normalizadas: ')
print(df[col_sensores_uteis].head())

Janela = 30

x = []
y = []

for motor in df['unit_number'].unique():
  dados_motor = df[df['unit_number'] == motor]

  sensores = dados_motor[col_sensores_uteis].values
  rul = dados_motor['RUL'].values

  for i in range(len(dados_motor) - Janela + 1):
    x.append(sensores[i : i + Janela])
    y.append(rul[i + Janela - 1])

x = np.array(x)
y = np.array(y)

print('Formato de X (janelas, ciclos, sensores): ', x.shape)
print('Formato de Y (alvos): ', y.shape)
print('\nExemplo - RUL alvo das 5 primeiras janelas', y[5])

from sklearn.model_selection import train_test_split

x_treino, x_val, y_treino, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print('Treino - x: ', x_treino.shape, '| y: ', y_treino.shape)
print('Validação - x: ', x_val.shape, '| y: ', y_val.shape)

RUL_MAX = 125.0

y_treino_norm = y_treino / RUL_MAX
y_val_norm = y_val / RUL_MAX

print('y_treino_norm: ', y_treino_norm.min(), 'a', y_treino_norm.max(), '(deve ser 0 a 1)')
print("y_val_norm:", y_val_norm.min(), "a", y_val_norm.max())


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

modelo = Sequential([
    LSTM(64, return_sequences=True, input_shape=(Janela,14)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

modelo.compile(
    loss='mse',
    optimizer='Adam',
    metrics=['mae']
)

modelo.summary()

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

historico = modelo.fit(
    x_treino, y_treino_norm,
    validation_data=(x_val, y_val_norm),
    epochs=100,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)


print('=== Alvo (y) ===')
print('y_treino - minimo: ', y_treino.min(), '| maximo: ', y_treino.max())

print("\n=== FEATURES (X) ===")
print("X_treino — mínimo:", x_treino.min(), "| máximo:", x_treino.max())
print("Algum valor NaN em X?", np.isnan(x_treino).any())
print("Algum valor NaN em y?", np.isnan(y_treino).any())

previsoes_teste = modelo.predict(x_val[:5])
print("Previsões do modelo:", previsoes_teste.flatten().round(1))
print("Valores reais:      ", y_val[:5])

print("y (janelamento):", y.min(), "a", y.max())
print("y_treino:", y_treino.min(), "a", y_treino.max())
print("y_val:", y_val.min(), "a", y_val.max())

# Testa o modelo QUE ACABOU de treinar, com variáveis corretas
prev_norm = modelo.predict(x_val)
prev_ciclos = prev_norm.flatten() * RUL_MAX

mae_ciclos = np.mean(np.abs(prev_ciclos - y_val))
print("MAE em ciclos reais:", round(mae_ciclos, 2))
print("\nPrevisões (ciclos):", prev_ciclos[:10].round(1))
print("Reais (ciclos):    ", y_val[:10])

df_teste = pd.read_csv('test_FD001.txt', sep= r'\s+', header=None, names=col_nomes)

rul_verdadeiro   = pd.read_csv('RUL_FD001.txt', header=None, names=['RUL_real'])
rul_verdadeiro = rul_verdadeiro['RUL_real'].values

df_teste = df_teste.drop(columns=col_remove)

df_teste[col_sensores_uteis] = scaler.transform(df_teste[col_sensores_uteis])

x_teste = []

for motor in df_teste['unit_number'].unique():
  dados_motor = df_teste[df_teste['unit_number'] == motor]
  sensores = dados_motor[col_sensores_uteis].values

  if len(sensores) >= Janela:
    ultima_janela = sensores[-Janela:]

  else:
    faltam = Janela - len(sensores)
    preenchimento = np.repeat(sensores[0:1], faltam, axis = 0)
    ultima_janela = np.vstack([preenchimento, sensores])

  x_teste.append(ultima_janela)

x_teste = np.array(x_teste)
print('Formato de x_teste (deve ser 100, ', Janela,', 14): ', x_teste.shape)
prev_teste_norm = modelo.predict(x_teste)
prev_teste = prev_teste_norm.flatten() * 125
prev_teste = np.clip(prev_teste, 0, 125)

rmse = np.sqrt(np.mean((prev_teste - rul_verdadeiro) ** 2))

mae = np.mean(np.abs(prev_teste - rul_verdadeiro))

print("\n===== RESULTADOS NO TESTE OFICIAL =====")
print("RMSE:", round(rmse, 2), "ciclos")
print("MAE: ", round(mae, 2), "ciclos")
print("\nExemplo — Previsões:", prev_teste[:10].round(1))
print("Exemplo — Reais:    ", rul_verdadeiro[:10])


# Pega a janela de teste do primeiro motor e inspeciona
print("Formato de uma janela de teste:", x_teste[0].shape, "(deve ser", Janela, ", 14)")
print("\nPrimeira janela de teste — primeiras 3 linhas (3 ciclos):")
print(x_teste[0][:3].round(3))
print("\nComparação: primeira janela de VALIDAÇÃO — primeiras 3 linhas:")
print(x_val[0][:3].round(3))

# Confirma que a validação funciona NESTA sessão (pós reinício)
prev_val = modelo.predict(x_val[:10]) * 125.0
print("Previsões VALIDAÇÃO (ciclos):", prev_val.flatten().round(1))
print("Reais VALIDAÇÃO (ciclos):    ", y_val[:10])
print("\n--- agora o teste ---")
prev_tst = modelo.predict(x_teste[:10]) * 125.0
print("Previsões TESTE (ciclos):", prev_tst.flatten().round(1))
print("Reais TESTE (ciclos):    ", rul_verdadeiro[:10])

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(historico.history['loss'], label='Treino')
plt.plot(historico.history['val_loss'], label='Validação')
plt.title('Curva de Aprendizado — Perda (MSE) por Época')
plt.xlabel('Época')
plt.ylabel('Perda (MSE, escala normalizada)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/content/grafico_curva_treino.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(7, 7))
plt.scatter(rul_verdadeiro, prev_teste, alpha=0.6, edgecolors='k', linewidths=0.5)
# linha diagonal (previsão perfeita)
lim_max = max(rul_verdadeiro.max(), prev_teste.max())
plt.plot([0, lim_max], [0, lim_max], 'r--', label='Previsão perfeita')
plt.title('RUL Previsto vs. RUL Real (Teste)')
plt.xlabel('RUL Real (ciclos)')
plt.ylabel('RUL Previsto (ciclos)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/content/grafico_previsto_vs_real.png', dpi=150, bbox_inches='tight')
plt.show()

# Ordena pelos valores reais, do menor pro maior
ordem = np.argsort(rul_verdadeiro)
real_ord = rul_verdadeiro[ordem]
prev_ord = prev_teste[ordem]

plt.figure(figsize=(12, 5))
plt.plot(real_ord, label='RUL Real', linewidth=2)
plt.plot(prev_ord, label='RUL Previsto', alpha=0.7)
plt.title('RUL Previsto vs. Real — 100 motores de teste (ordenados)')
plt.xlabel('Motores (ordenados por RUL real crescente)')
plt.ylabel('RUL (ciclos)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/content/grafico_previsoes_ordenadas.png', dpi=150, bbox_inches='tight')
plt.show()