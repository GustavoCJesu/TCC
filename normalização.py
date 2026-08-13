import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from scipy.signal import savgol_filter

def suavizar(df, col_sensores, window=7, poly=2):
    df = df.copy()
    def f(g):
        for s in col_sensores:
            n = len(g)
            w = window if n >= window else (n if n % 2 == 1 else n - 1)
            if w > poly:
                g[s] = savgol_filter(g[s].values, w, poly)
        return g
    return df.groupby('unit_number', group_keys=False).apply(f)

def normalizacao(df, df_teste, janela=30, RUL_MAX=125):
    col = [c for c in df.columns if c.startswith('sensor_') or c.startswith('op_setting_')]
    desvios = df[col].std()
    col_remove = desvios[desvios == 0].index
    df = df.drop(columns=col_remove)
    df_teste = df_teste.drop(columns=col_remove)

    col_sensores = [c for c in df.columns if c.startswith('sensor_')]

    # >>> SUAVIZAÇÃO ENTRA AQUI: depois do corte, ANTES da normalização <
    df = suavizar(df, col_sensores, window=7, poly=2)
    df_teste = suavizar(df_teste, col_sensores, window=7, poly=2)


    motores = df['unit_number'].unique()
    m_treino, m_val = train_test_split(motores, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    # fit SÓ nos motores de treino
    mask_tr_df = df['unit_number'].isin(m_treino)
    scaler.fit(df.loc[mask_tr_df, col_sensores])

    df[col_sensores] = scaler.transform(df[col_sensores])
    df_teste[col_sensores] = scaler.transform(df_teste[col_sensores])

    x, y, ids = [], [], []
    
    for motor in df['unit_number'].unique():
        
        dados_motor = df[df['unit_number'] == motor]
        sensores = dados_motor[col_sensores].values
        
        rul = dados_motor['RUL'].values
        
        for i in range(len(dados_motor) - janela + 1):
            x.append(sensores[i:i+janela])
            y.append(rul[i+janela-1])
            ids.append(motor)
            
    x, y, ids = np.array(x), np.array(y), np.array(ids)

    mask_tr = np.isin(ids, m_treino)
    mask_val = np.isin(ids, m_val)
    x_treino, y_treino = x[mask_tr], y[mask_tr]
    x_val, y_val = x[mask_val], y[mask_val]

    y_treino_norm = y_treino / RUL_MAX
    y_val_norm = y_val / RUL_MAX

    # teste (igual ao seu)
    x_teste = []
    for motor in df_teste['unit_number'].unique():
        
        dados_motor = df_teste[df_teste['unit_number'] == motor]
        sensores = dados_motor[col_sensores].values
        
        if len(sensores) >= janela:
            ultima_janela = sensores[-janela:]
            
        else:
            faltam = janela - len(sensores)
            preenchimento = np.repeat(sensores[0:1], faltam, axis=0)
            ultima_janela = np.vstack([preenchimento, sensores])
            
        x_teste.append(ultima_janela)
        
    x_teste = np.array(x_teste)

    return x_treino, x_val, y_treino_norm, y_val_norm, scaler, col_sensores, col_remove, y_val, x_teste
    