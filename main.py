from dados import coletaDados
from normalização import normalizacao
from modelo import modelo
from graficos import geraGraficos


def pergunta(texto, padrao):
    try:
        resposta = input(f'{texto} [{padrao}]: ').strip()
    except EOFError:
        return padrao
    return int(resposta) if resposta else padrao


janela = pergunta('Tamanho de janela', 25)
teto = pergunta('Teto Maximo do RUL', 150)
epocas = pergunta('Número de épocas', 30)
suavizacao = pergunta('Suavizar sensores, 1 sim 0 não', 0)
limiar_auto = pergunta('Escolher limiar da triagem na validação, 1 sim 0 não', 0)
margem = pergunta('Margem de RUL acima de 100 no treino do modelo 2', 0)
empilhada = pergunta('Rede LSTM empilhada, 1 sim 0 não', 0)
devolve = pergunta('Devolver a Saudável se o modelo 2 prever RUL >= 100, 1 sim 0 não', 0)
peso_degradado = pergunta('Peso do erro em motor degradado no modelo 1', 1)
n_modelos = pergunta('Redes por modelo, média do ensemble', 1)
semente = pergunta('Semente aleatória', 42)

df, df_teste, rul_verdadeiro = coletaDados(teto)

x_treino, x_val, y_treino_norm, y_val_norm, scaler, col_sensores, col_remove, y_val, x_teste = normalizacao(
    df, df_teste, janela, teto, suavizacao=bool(suavizacao))

print(f'\nJanelas de treino: {len(x_treino)} | validação: {len(x_val)} | '
      f'motores de teste: {len(x_teste)}')
print(f'Features por passo: {x_treino.shape[2]}')

historicos, rul_verdadeiro, previsoes, metricas = modelo(
    janela, teto, x_treino, x_val, y_treino_norm, y_val_norm, y_val,
    df, x_teste, rul_verdadeiro, epocas=epocas, limiar_auto=bool(limiar_auto),
    semente=semente, n_modelos=n_modelos, empilhada=bool(empilhada),
    margem=margem, devolve=bool(devolve), peso_degradado=peso_degradado)

geraGraficos(historicos, rul_verdadeiro, previsoes, janela, teto, metricas)
