


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np

def geraGraficos(historico, rul_verdadeiro, prev_teste):
    
    plt.figure(figsize=(10, 5))
    plt.plot(historico.history['loss'], label='Treino')
    plt.plot(historico.history['val_loss'], label='Validação')
    plt.title('Curva de Aprendizado — Perda (MSE) por Época')
    plt.xlabel('Época')
    plt.ylabel('Perda (MSE, escala normalizada)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('grafico_curva_treino.png', dpi=150, bbox_inches='tight')
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
    plt.savefig('grafico_previsto_vs_real.png', dpi=150, bbox_inches='tight')
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
    plt.savefig('grafico_previsoes_ordenadas.png', dpi=150, bbox_inches='tight')
    plt.show()
