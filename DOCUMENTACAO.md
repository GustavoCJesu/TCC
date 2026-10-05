# Documentação do programa

Previsão de vida útil restante (RUL) de motores turbofan com o dataset NASA C-MAPSS (subconjunto FD001), usando dois modelos LSTM em cascata.

O programa parte de uma versão crua (baseline) e oferece técnicas de melhoria que podem ser ligadas uma a uma. Com tudo desligado (Enter em todas as perguntas), ele reproduz o modelo cru.

## 1. Visão geral

| Modelo | Pergunta | Treina com | Saída |
|---|---|---|---|
| Modelo 1 (triagem) | O motor ainda está saudável? | Todas as janelas | P(não saudável) |
| Modelo 2 (diagnóstico) | Quantos ciclos restam? | Janelas com RUL < 100 + margem | RUL previsto |

| Estado | RUL | Quem decide |
|---|---|---|
| Saudável | ≥ 100 | Modelo 1 |
| Atenção | 50 a 99 | Modelo 2 (RUL previsto ≥ 50) |
| Perigoso | < 50 | Modelo 2 (RUL previsto < 50) |

```
leituras ──► Modelo 1 ──► P(não saudável) < limiar ──► SAUDÁVEL (RUL não estimado)
                 │
                 └─ >= limiar ──► Modelo 2 ──► RUL previsto
                                               ├─ >= 100 e "devolve" ligado ──► SAUDÁVEL
                                               ├─ < 50  ──► PERIGOSO
                                               └─ >= 50 ──► ATENÇÃO
```

Ordem em [main.py](main.py): `coletaDados` → `normalizacao` → `modelo` → `geraGraficos`. Os limites 50 e 100 ficam em `LIMITES_SAUDE` ([avaliacao.py](avaliacao.py)) e devem ser confirmados com o orientador.

---

## 2. Módulos

### 2.1 [main.py](main.py)
Pergunta os parâmetros (todos com padrão desligado) e executa o pipeline. `pergunta()` lê um inteiro e devolve o padrão se a entrada acabar.

| Pergunta | Padrão | Efeito |
|---|---|---|
| Tamanho de janela | 25 | Ciclos por janela |
| Teto do RUL | 150 | Limite do RUL e escala da normalização |
| Número de épocas | 30 | Passadas completas no treino |
| Suavizar sensores | 0 | Filtro Savitzky-Golay causal |
| Limiar da triagem na validação | 0 | Escolhe o limiar do modelo 1 na validação |
| Margem de RUL acima de 100 | 0 | Modelo 2 treina com RUL < 100 + margem |
| Rede LSTM empilhada | 0 | Duas camadas LSTM (64 e 32) |
| Devolver a Saudável | 0 | Modelo 2 prevendo RUL ≥ 100 devolve o motor a Saudável |
| Peso do erro em degradado | 1 | Peso da classe "não saudável" no modelo 1 |
| Redes por modelo | 1 | Ensemble: média de N redes |
| Semente | 42 | Inicialização aleatória |

### 2.2 [dados.py](dados.py)
`coletaDados(RUL_TETO)`: lê treino, teste e RUL real; calcula o RUL por ciclo e o limita ao teto, pois no início da vida os sensores não mostram degradação.

### 2.3 [normalização.py](normalização.py)
- `suavizar(df, col_sensores, window, poly)`: filtro Savitzky-Golay **causal** por motor (janela 7, grau 2). Cada ponto usa só os ciclos anteriores, como na operação real. Os 6 primeiros ciclos ficam sem alteração. A versão não causal usa pontos futuros e dá ao treino uma informação que o teste não tem.
- `normalizacao(df, df_teste, janela, RUL_MAX, suavizacao)`: remove colunas constantes (ficam 17 sensores), suaviza se pedido, separa 80% treino e 20% validação **por motor** (semente 42), ajusta o MinMaxScaler só no treino, monta janelas deslizantes e uma janela final por motor de teste.

### 2.4 [modelo.py](modelo.py)
- `constroi_triagem`, `constroi_diagnostico`: LSTM de 64 unidades (ou 64 e 32 se empilhada). O modelo 1 termina em sigmoide com entropia cruzada; o modelo 2 termina em saída linear com MSE. Otimizador Adam.
- `treina_ensemble`: treina N redes com sementes consecutivas, com lotes de 256, e aceita pesos de classe.
- `previsoes_brutas`: média das redes, devolvendo P(não saudável) e RUL para todas as janelas.
- `decide(prob, rul, limiar, devolve)`: aplica a regra da cascata e devolve motores enviados, classes e RUL.
- `recall_medio`, `escolhe_limiar`: testam limiares de 0,20 a 0,80 (e 0,10 e 0,15 com devolução) nas janelas de validação e escolhem o de maior recall médio dos três estados; empate vai para o mais próximo de 0,5.
- `modelo(...)`: orquestra treino, escolha do limiar, inferência, métricas e laudo.

**Por quê:**
- **Cascata:** o modelo 2 só vê motores degradados, então dedica toda a capacidade ao RUL nessa faixa.
- **Estado pelo RUL previsto:** o modelo 2 tem uma saída só; a classe vem do corte em 50.
- **Margem:** sem ela o modelo 2 nunca viu motores perto da fronteira e subestima o RUL dos falsos positivos.
- **Limiar na validação:** a escolha não usa o teste.
- **Devolução:** permite baixar o limiar sem que os falsos alarmes virem Atenção.
- **Peso do erro:** liberar um motor degradado custa mais que um alarme falso.
- **Ensemble:** reduz a variação por inicialização.

### 2.5 [avaliacao.py](avaliacao.py)
`score_nasa`, `avalia`, `calcula_faixas`, `classe_saude`, `avalia_triagem`, `avalia_saude`, `avalia_intervalo` (hoje sem uso, reservada para quantis). As métricas de RUL valem só para os motores enviados ao modelo 2, e o RUL real é limitado ao teto.

### 2.6 [graficos.py](graficos.py)
Cinco imagens: curvas de aprendizado (um painel por modelo, com faixa de mínimo a máximo quando há ensemble), previsto contra real, previsões ordenadas, erro por faixa e matriz de confusão 3×3.

### 2.7 [experimentos.py](experimentos.py)
Roda configurações com 5 sementes (42 a 46) e grava uma linha por execução em `resultados_experimentos.csv`. Pula execuções já feitas, o que permite retomar.

```
python3 experimentos.py                 # todas as configurações
python3 experimentos.py cru margem20    # só as indicadas
python3 experimentos.py resumo          # média e desvio por configuração
```

Configurações: cru, suavizacao, janela30, empilhada, margem20, limiar_auto, peso2, peso3, devolve, combinado, devolve_peso2, combinado_ensemble, devolve_peso2_ensemble.

**Por quê:** com 100 motores de teste, uma só execução não distingue melhoria de variação. Cada técnica é medida isoladamente contra a base, com as mesmas sementes e a mesma divisão de motores, o que permite comparar par a par.

---

## 3. Resultados (5 sementes, média ± desvio, teste com 100 motores)

| Configuração | Acurácia | Recall Saudável | Recall Atenção | Recall Perigoso | RMSE |
|---|---|---|---|---|---|
| Cru | 81,0 ± 3,2 | 91,8 | 57,2 | 97,3 | 13,8 |
| Suavização | 79,0 ± 1,6 | 92,4 | 51,1 | 97,3 | 13,6 |
| Janela 30 | 78,2 ± 2,4 | 88,2 | 52,2 | 98,0 | 15,7 |
| Empilhada | 79,0 ± 2,0 | 90,6 | 54,4 | 95,3 | 13,7 |
| Margem 20 | 81,4 ± 2,9 | 91,8 | 58,3 | 97,3 | 12,5 |
| Limiar auto | 82,8 ± 2,7 | 87,1 | 66,7 | 97,3 | 15,1 |
| Peso 2 + margem 20 | 82,4 ± 3,6 | 82,4 | 70,0 | 97,3 | 13,6 |
| Peso 3 + margem 20 | 80,4 ± 4,5 | 70,0 | 76,1 | 97,3 | 14,0 |
| Limiar auto + margem 20 | 83,2 ± 0,8 | 87,1 | 67,8 | 97,3 | 12,8 |
| + devolução | 84,0 ± 2,0 | 85,3 | 71,7 | 97,3 | 13,2 |
| + devolução + peso 2 | 84,8 ± 1,8 | 87,1 | 72,2 | 97,3 | 13,1 |
| Limiar auto + margem 20 + ensemble 3 | 85,4 ± 0,9 | 89,4 | 72,2 | 96,7 | 12,0 |
| + devolução + peso 2 + ensemble 3 | 85,8 ± 1,1 | 88,8 | 73,9 | 96,7 | 12,5 |

Leitura:
- Em todas as 65 execuções, nenhum motor Perigoso foi liberado como Saudável.
- A margem é a única melhoria consistente no RUL (RMSE melhor nas 5 sementes).
- Limiar, peso e devolução deslocam a troca entre recall de Atenção e de Saudável. Não são ganhos puros isoladamente.
- O ensemble é o que mais reduz a variação entre sementes (3,2 para 0,9 ponto de acurácia).
- Suavização, janela 30 e rede empilhada não ajudaram nesta base.
- A configuração com devolução e peso 2 e ensemble não se distingue da versão sem eles (diferença de 0,4 ponto, dentro do ruído), então a versão mais simples é igualmente defensável.

## 4. Como executar

```
cd "/home/sempher/Área de Trabalho/Gustavo/TCC"
python3 main.py
```

Configuração do melhor resultado: janela 25, teto 150, 30 épocas, suavização 0, limiar automático 1, margem 20, empilhada 0, devolução 1, peso 2, redes 3, semente 42.
