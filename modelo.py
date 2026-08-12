from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd


def modelo(janela, teto, x_treino, x_val, y_treino_norm, y_val_norm, y_val, df, x_teste, rul_verdadeiro):
    
    modelo = Sequential([
    LSTM(64, return_sequences=True, input_shape=(janela, x_treino.shape[2])),
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
        verbose = 1
    )
    
    prev_norm = modelo.predict(x_val)
    prev_ciclos = prev_norm.flatten() * teto
    
    
    print('\n========VALORES DO TREINO========\n')
    
    mae_ciclos = np.mean(np.abs(prev_ciclos - y_val))
    print("MAE em ciclos reais:", round(mae_ciclos, 2))
    print("\nPrevisões (ciclos):", prev_ciclos[:10].round(1))
    print("Reais (ciclos):    ", y_val[:10])
    print(y_val.min(), y_val.max(), np.unique(y_val)[:20])
    print('\n========================\n')
    
    #Inicio do teste do modelo no arquivo test_FD001.txt
    
    prev_teste_norm = modelo.predict(x_teste)
    prev_teste = prev_teste_norm.flatten() * teto
    prev_teste = np.clip(prev_teste,0, teto)
    
    rmse = np.sqrt(np.mean((prev_teste - rul_verdadeiro) ** 2))
    mae = np.mean(np.abs(prev_teste - rul_verdadeiro))
    
    print("\n===== RESULTADOS NO TESTE OFICIAL =====")
    print("RMSE:", round(rmse, 2), "ciclos")
    print("MAE: ", round(mae, 2), "ciclos")
    print("\nExemplo — Previsões:", prev_teste[:10].round(1))
    print("Exemplo — Reais:    ", rul_verdadeiro[:10])
    
    return historico, rul_verdadeiro, prev_teste
    
    