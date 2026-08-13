import numpy as np
import pandas as pd


def coletaDados(RUL_TETO = 125):
    
    
    #Definição dos nomes das colunas
    col_nomes = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    col_nomes += [f'sensor_{i}' for i in range(1, 22)]


    #Leitura do arquivo de treino, usando 'espaços' como separador, indicando que não temos um header, e que vamos nomear cada coluna com as colunas inseridas na variavel 'col_nomes'
    df = pd.read_csv('CMAPSSData/train_FD001.txt', sep=r'\s+', header=None, names=col_nomes)
    df_teste = pd.read_csv('CMAPSSData/test_FD001.txt', sep=r'\s+', header=None, names=col_nomes)

    rul_verdadeiro = pd.read_csv('CMAPSSData/RUL_FD001.txt', header=None, names=['RUL_real'])
    rul_verdadeiro = rul_verdadeiro['RUL_real'].values

    #Armazenando o maior ciclo de cada motor
    ciclo_max = df.groupby('unit_number')['time_in_cycles'].max()


    #print(ciclo_max)


    #Define a coluna 'ciclo_max' com o valor total da vida do motor
    df['ciclo_max'] = df['unit_number'].map(ciclo_max)


    #Define a coluna 'RUL' com a quantidade de ciclos restantes da vida do motor, subitraindo o 'ciclo_max' com a coluna 'time_in_cycles', sendo o resultado, o tempo de vida restante do motor
    df['RUL'] = df['ciclo_max'] - df['time_in_cycles']

    # print('RUL antes do limte do teto:\n', df['RUL'])


    #Comando clip(upper=RUL_TETO) vai limitar o malor maximo de RUL ao valor informado Ex: RUL_TETO = 120, df['RUL'] = 200 passa a ser 120
    df['RUL'] = df['RUL'].clip(upper=RUL_TETO)
    
    
    # print(f'RUL depois do limte do teto deve ser {RUL_TETO}:\n', df['RUL'])

    #Comando .drop(columns='ciclo_max') para apagar a coluna 'ciclo_max' que no momento virou lixo pois seria varias linhas com o mesmo valor
    df = df.drop(columns='ciclo_max')
    
    return df, df_teste, rul_verdadeiro