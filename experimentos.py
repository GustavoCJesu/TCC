import contextlib
import csv
import io
import os
import sys
import time

import numpy as np

from dados import coletaDados
from normalização import normalizacao
from modelo import modelo

TETO = 150
EPOCAS = 30
SEMENTES = [42, 43, 44, 45, 46]
ARQUIVO = 'resultados_experimentos.csv'
CAMPOS = ['config', 'semente', 'limiar', 'val_acuracia', 'val_recall_medio',
          'acuracia', 'rec_saudavel', 'rec_atencao', 'rec_perigoso',
          'graves', 'enviados', 'tri_recall', 'tri_precisao', 'rmse', 'mae',
          'score', 'segundos']

CONFIGS = {
    'cru': dict(),
    'suavizacao': dict(suavizacao=True),
    'limiar_auto': dict(limiar_auto=True),
    'margem20': dict(margem=20),
    'janela30': dict(janela=30),
    'empilhada': dict(empilhada=True),
    'combinado': dict(limiar_auto=True, margem=20),
    'combinado_ensemble': dict(limiar_auto=True, margem=20, n_modelos=3),
    'peso2': dict(peso_degradado=2.0, margem=20),
    'peso3': dict(peso_degradado=3.0, margem=20),
    'devolve': dict(limiar_auto=True, margem=20, devolve=True),
    'devolve_peso2': dict(limiar_auto=True, margem=20, devolve=True,
                          peso_degradado=2.0),
    'devolve_peso2_ensemble': dict(limiar_auto=True, margem=20, devolve=True,
                                   peso_degradado=2.0, n_modelos=3),
}

OPCOES_DADOS = ('janela', 'suavizacao')
cache = {}


def dados(janela, suavizacao):
    chave = (janela, suavizacao)
    if chave not in cache:
        df, df_teste, rul = coletaDados(TETO)
        with contextlib.redirect_stdout(io.StringIO()):
            partes = normalizacao(df, df_teste, janela, TETO,
                                  suavizacao=suavizacao)
        cache[chave] = (df, rul, partes)
    return cache[chave]


def roda(nome, semente):
    opcoes = dict(CONFIGS[nome])
    janela = opcoes.pop('janela', 25)
    suavizacao = opcoes.pop('suavizacao', False)
    df, rul, partes = dados(janela, suavizacao)
    x_tr, x_val, y_tr, y_val_n, _, _, _, y_val, x_te = partes

    inicio = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, previsoes, m = modelo(
            janela, TETO, x_tr, x_val, y_tr, y_val_n, y_val, df, x_te, rul,
            epocas=EPOCAS, semente=semente, verbose=0, **opcoes)

    rec = m['saude']['recall']
    matriz = m['saude']['matriz']
    return dict(
        config=nome, semente=semente, limiar=m['limiar'],
        val_acuracia=m['validacao']['acuracia'],
        val_recall_medio=m['validacao']['recall_medio'],
        acuracia=m['saude']['acuracia'],
        rec_saudavel=rec[0], rec_atencao=rec[1], rec_perigoso=rec[2],
        graves=int(matriz[0, 2] + matriz[2, 0]),
        enviados=int(previsoes['enviado'].sum()),
        tri_recall=m['triagem']['recall'], tri_precisao=m['triagem']['precisao'],
        rmse=m['rmse'], mae=m['mae'], score=m['score'],
        segundos=round(time.time() - inicio, 1))


def grava(linha):
    novo = not os.path.exists(ARQUIVO)
    with open(ARQUIVO, 'a', newline='') as f:
        escritor = csv.DictWriter(f, CAMPOS)
        if novo:
            escritor.writeheader()
        escritor.writerow(linha)


def feitos():
    if not os.path.exists(ARQUIVO):
        return set()
    with open(ARQUIVO, newline='') as f:
        return {(l['config'], int(l['semente'])) for l in csv.DictReader(f)}


def resumo():
    with open(ARQUIVO, newline='') as f:
        linhas = list(csv.DictReader(f))
    colunas = ['val_acuracia', 'val_recall_medio', 'acuracia', 'rec_atencao',
               'rec_saudavel', 'rec_perigoso', 'graves', 'rmse', 'score']
    print(f'{"config":<20}{"n":>3}' + ''.join(f'{c:>20}' for c in colunas))
    for nome in CONFIGS:
        sel = [l for l in linhas if l['config'] == nome]
        if not sel:
            continue
        celulas = []
        for c in colunas:
            v = np.array([float(l[c]) for l in sel])
            celulas.append(f'{v.mean():9.3f} ±{v.std(ddof=1) if len(v) > 1 else 0:6.3f}')
        print(f'{nome:<20}{len(sel):>3}' + ''.join(f'{c:>20}' for c in celulas))


if __name__ == '__main__':
    argumentos = sys.argv[1:]
    if argumentos == ['resumo']:
        resumo()
        sys.exit()

    nomes = argumentos or list(CONFIGS)
    ja = feitos()
    for nome in nomes:
        for semente in SEMENTES:
            if (nome, semente) in ja:
                continue
            linha = roda(nome, semente)
            grava(linha)
            print(f'{nome} semente={semente} acuracia={linha["acuracia"]:.2f} '
                  f'rec_atencao={linha["rec_atencao"]:.2f} '
                  f'limiar={linha["limiar"]:.2f} ({linha["segundos"]}s)',
                  flush=True)
