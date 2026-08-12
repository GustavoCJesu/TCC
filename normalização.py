import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def normalizacao(df, df_teste, janela = 30, RUL_MAX = 125):
    
    scaler = MinMaxScaler()
    
    #Laço for para percorrer todas as colunas que começam com 'sensor_', coletando todos os sensores e armazenando em um array
    col = [c for c in df.columns if c.startswith('sensor_') or c.startswith('op_setting_')]
    
    #Calcula o desvio padrão de todas as colunas
    desvios = df[col].std()
    
    #Guarda as colunas cujo o desvio padrão é igual a 0
    col_remove = desvios[desvios == 0].index
    
    #Usa as colunas guardadas em col_remove pra remover do DataFrame
    df = df.drop(columns=col_remove)
    df_teste = df_teste.drop(columns=col_remove)
    
    #Guarda todos os sensores do DataFrame
    col_sensores = [c for c in df.columns if c.startswith('sensor_')]
    
    #Usando MinMaxSacaler, normalizamos os valores das colunas armazenadas em 'col_sensores' e armazenamos em si mesma, deixando os valores de cada linhas entre o intervalo de 1 e 0
    df[col_sensores] = scaler.fit_transform(df[col_sensores])
    df_teste[col_sensores] = scaler.transform(df_teste[col_sensores])
    
    
    #Criação dos arrays para armazenar os valores
    x = []
    y = []
    ids = []
    
    #Nota: Olhar melhor essa parte para compreensão
    #Usando o comando unique, pegamos somente um motor de cada vez e salvamos todas as informações dele em 'motor'
    for motor in df['unit_number'].unique():
        
        #Salva todos os dados do motor
        dados_motor = df[df['unit_number'] == motor]
        
        #Separa somente os dados dos sensores e os salva como valores brutos, ignorando o tipo gerado pelo pandas
        sensores = dados_motor[col_sensores].values
        
        #Salva o RUL do motor tambem como valores brutos ignorando o tipo gerado pelo pandas
        rul = dados_motor['RUL'].values
        
        for i in range(len(dados_motor) - janela + 1):
            x.append(sensores[i : i + janela])
            y.append(rul[i + janela - 1])
            ids.append(motor)
            
    x = np.array(x)
    y = np.array(y)
    ids = np.array(ids)
    
    
    motores = df['unit_number'].unique()
    m_treino, m_val = train_test_split(motores, test_size = 0.2, random_state = 42)
    
    
    mask_tr = np.isin(ids, m_treino)
    mask_val = np.isin(ids, m_val)
    
    
    x_treino, y_treino = x[mask_tr], y[mask_tr]
    x_val, y_val = x[mask_val], y[mask_val]
    
    
    y_treino_norm = y_treino / RUL_MAX
    y_val_norm = y_val / RUL_MAX
    
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
    