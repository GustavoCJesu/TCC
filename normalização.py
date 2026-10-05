import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from scipy.signal import savgol_coeffs


def suavizar(df, col_sensores, window=7, poly=2):
    coef = savgol_coeffs(window, poly, pos=window - 1)

    def filtra(serie):
        v = serie.values
        suave = np.convolve(v, coef, mode='full')[:len(v)]
        suave[:window - 1] = v[:window - 1]
        return pd.Series(suave, index=serie.index)

    df = df.copy()
    for s in col_sensores:
        df[s] = df.groupby('unit_number')[s].transform(filtra)
    return df


def normalizacao(df, df_teste, janela=30, RUL_MAX=125, suavizacao=False):
    col = [c for c in df.columns if c.startswith('sensor_') or c.startswith('op_setting_')]
    desvios = df[col].std()
    col_remove = desvios[desvios == 0].index
    df = df.drop(columns=col_remove)
    df_teste = df_teste.drop(columns=col_remove)

    col_sensores = [c for c in df.columns if c.startswith('sensor_')]

    if suavizacao:
        df = suavizar(df, col_sensores)
        df_teste = suavizar(df_teste, col_sensores)

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

    # Teste: cada motor vira UMA janela, a mais recente disponível.
    x_teste = []
    motores_preenchidos = []

    for motor in df_teste['unit_number'].unique():

        dados_motor = df_teste[df_teste['unit_number'] == motor]
        sensores = dados_motor[col_sensores].values

        if len(sensores) >= janela:
            ultima_janela = sensores[-janela:]

        else:
            # A série é mais curta que a janela. Repetimos a primeira leitura
            # para trás, assumindo que o motor estava em estado semelhante
            # antes do início da medição. Isso FABRICA dados: os ciclos
            # preenchidos não foram observados. No FD001 a menor série de
            # teste tem 31 ciclos, então isso só dispara com janela >= 32.
            faltam = janela - len(sensores)
            preenchimento = np.repeat(sensores[0:1], faltam, axis=0)
            ultima_janela = np.vstack([preenchimento, sensores])
            motores_preenchidos.append((int(motor), len(sensores), faltam))

        x_teste.append(ultima_janela)

    x_teste = np.array(x_teste)

    if motores_preenchidos:
        print(f'\n[ATENÇÃO] janela={janela} é maior que a série de '
              f'{len(motores_preenchidos)} motor(es) de teste. '
              f'Ciclos foram preenchidos artificialmente:')
        for m, n, f in motores_preenchidos[:10]:
            print(f'  motor {m}: {n} ciclos reais + {f} preenchidos')
        print('  As métricas desses motores não são confiáveis. '
              'Considere reduzir a janela.\n')

    return x_treino, x_val, y_treino_norm, y_val_norm, scaler, col_sensores, col_remove, y_val, x_teste
    