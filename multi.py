# ============================================================
# PROBLEMA DE DIMENSIONAMENTO DE LOTES MULTIESTÁGIOS
# ============================================================
# Objetivo: Minimizar o custo total de produção e estocagem
#           ao longo de T períodos, considerando múltiplos itens
#           e recursos limitados, com consumo interno entre itens.
# ============================================================

import numpy as np
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD

# ------------------------------------------------------------
# Função: ler_arquivo_multi
# ------------------------------------------------------------
# Lê os dados do problema de um arquivo texto e retorna todas
# as matrizes e vetores necessários para o modelo.
# ------------------------------------------------------------
def ler_arquivo_multi(problema):
    # Abre o arquivo e remove linhas vazias
    with open(problema, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    idx = 0  # índice de leitura de linhas

    # Primeira linha: T, n, K
    T, n, K = map(int, lines[idx].split()); idx += 1

    # ----------------------------------------
    # Demanda de cada item em cada período (n x T)
    # d[j][t] = demanda do item j no período t
    # ----------------------------------------
    d = np.array([list(map(float, lines[idx + j].split())) for j in range(n)])
    idx += n

    # ----------------------------------------
    # Custos de produção (n x T)
    # c[j][t] = custo de produzir o item j no período t
    # ----------------------------------------
    c = np.array([list(map(float, lines[idx + j].split())) for j in range(n)])
    idx += n

    # ----------------------------------------
    # Custos de estocagem (n x T)
    # h[j][t] = custo de estocar o item j no final do período t
    # ----------------------------------------
    h = np.array([list(map(float, lines[idx + j].split())) for j in range(n)])
    idx += n

    # ----------------------------------------
    # Matriz B (n x n)
    # B[j][i] = quantidade do item j necessária para
    # produzir 1 unidade do item i
    # ----------------------------------------
    B = np.array([list(map(float, lines[idx + j].split())) for j in range(n)])
    idx += n

    # ----------------------------------------
    # Matriz de consumo de recursos (K x n)
    # Rcons[k][j] = quantidade do recurso k usada
    # para produzir 1 unidade do item j
    # ----------------------------------------
    Rcons = np.array([list(map(float, lines[idx + k].split())) for k in range(K)])
    idx += K

    # ----------------------------------------
    # Disponibilidade de recursos (K x T)
    # Rcap[k][t] = capacidade disponível do recurso k
    # no período t
    # ----------------------------------------
    Rcap = np.array([list(map(float, lines[idx + k].split())) for k in range(K)])
    idx += K

    # ----------------------------------------
    # Estoque inicial (opcional)
    # I0[j] = estoque inicial do item j
    # ----------------------------------------
    I0 = np.zeros(n) # vetor de zeros se não fornecido
    if idx < len(lines): #verifica se há mais linhas a serem lidas
        parts = lines[idx].split() #pega a linha atual que não foi lida e separa em partes
        if len(parts) == n: #verifica se o número de partes é igual ao número de itens ou seja se cada item tem um estoque inicial
            I0 = np.array(list(map(float, parts))) #converte as partes para float 
            idx += 1 #incrementa o índice para a próxima linha

    # Retorna todos os parâmetros do problema
    return T, n, K, d, c, h, B, Rcons, Rcap, I0

# ------------------------------------------------------------
# Função: otimizar_multi
# ------------------------------------------------------------
# Constrói e resolve o modelo de dimensionamento multiestágio
# usando o solver CBC (via PuLP).
# ------------------------------------------------------------
def otimizar_multi(T, n, K, d, c, h, B, Rcons, Rcap, I0=None, use_integer=False):
    # Se nenhum estoque inicial for informado, assume zero
    if I0 is None:
        I0 = np.zeros(n)

    # Criação do problema de otimização (minimização)
    prob = LpProblem("Dimensionamento_Multiestagio", LpMinimize)

    # Dicionários para armazenar variáveis de decisão
    x = {}  # produção do item j no período t
    I = {}  # estoque do item j no final do período t

    BIG = 1e9  # constante grande (não usada, mas comum em modelos)

    # --------------------------------------------------------
    # Criação das variáveis x_{j,t} e I_{j,t}
    # --------------------------------------------------------
    for j in range(n):
        for t in range(1, T+1):
            name_x = f"x_{j+1}_{t}"
            name_I = f"I_{j+1}_{t}"

            # Define tipo das variáveis (inteiras ou contínuas)
            if use_integer:
                x[(j,t)] = LpVariable(name_x, lowBound=0, cat='Integer')
                I[(j,t)] = LpVariable(name_I, lowBound=0, cat='Integer')
            else:
                x[(j,t)] = LpVariable(name_x, lowBound=0)
                I[(j,t)] = LpVariable(name_I, lowBound=0)

    # --------------------------------------------------------
    # Função Objetivo:
    # Minimizar custo de produção + custo de estocagem
    # --------------------------------------------------------
    prob += lpSum([c[j,t-1] * x[(j,t)] for j in range(n) for t in range(1, T+1)]) \
          + lpSum([h[j,t-1] * I[(j,t)] for j in range(n) for t in range(1, T+1)])

    # --------------------------------------------------------
    # Restrições de Balanço de Estoque por item e período:
    #
    # x_{j,t} + I_{j,t-1} = d_{j,t} + Σ_i b_{j,i} * x_{i,t} + I_{j,t}
    #
    # Interpretação:
    # (produção + estoque anterior) deve atender
    # (demanda + consumo interno + estoque final)
    # --------------------------------------------------------
    for j in range(n):
        for t in range(1, T+1):
            # Estoque do período anterior (ou inicial no t=1)
            prev_I = I0[j] if t == 1 else I[(j, t-1)]

            # Consumo interno do item j para produção de outros itens
            consumo_interno = lpSum([B[j, i] * x[(i, t)] for i in range(n)])

            # Balanço de material
            prob += x[(j,t)] + prev_I - I[(j,t)] == d[j, t-1] + consumo_interno, \
                    f"balanco_item{j+1}_t{t}"

    # --------------------------------------------------------
    # Restrições de Capacidade dos Recursos:
    #
    # Σ_j r_{k,j} * x_{j,t} ≤ R_{k,t}
    #
    # Interpretação:
    # O consumo total de cada recurso em cada período
    # não pode ultrapassar a disponibilidade.
    # --------------------------------------------------------
    for k in range(K):
        for t in range(1, T+1):
            prob += lpSum([Rcons[k, j] * x[(j, t)] for j in range(n)]) <= Rcap[k, t-1], \
                    f"recurso_{k+1}_t{t}"

    # --------------------------------------------------------
    # Resolução do modelo com o solver CBC
    # --------------------------------------------------------
    prob.solve(PULP_CBC_CMD(msg=1))  # msg=1 exibe o log de solução

    # --------------------------------------------------------
    # Coleta dos resultados (valores das variáveis)
    # --------------------------------------------------------
    x_sol = np.zeros((n, T))
    I_sol = np.zeros((n, T))

    for j in range(n):
        for t in range(1, T+1):
            valx = x[(j,t)].varValue  # valor ótimo de x_{j,t}
            vali = I[(j,t)].varValue  # valor ótimo de I_{j,t}
            x_sol[j, t-1] = 0.0 if valx is None else float(valx)
            I_sol[j, t-1] = 0.0 if vali is None else float(vali)

    # Cálculo do custo total ótimo
    obj = sum(c[j,t] * x_sol[j,t] for j in range(n) for t in range(T)) + \
          sum(h[j,t] * I_sol[j,t] for j in range(n) for t in range(T))

    # Retorna solução ótima
    return x_sol, I_sol, obj


# ------------------------------------------------------------
# BLOCO PRINCIPAL - Exemplo de execução
# ------------------------------------------------------------
if __name__ == "__main__":
    # Leitura dos dados do arquivo de entrada
    T, n, K, d, c, h, B, Rcons, Rcap, I0 = ler_arquivo_multi("problema_multi.txt")

    # Chamada da função de otimização
    x_sol, I_sol, obj = otimizar_multi(T, n, K, d, c, h, B, Rcons, Rcap, I0, use_integer=False)

    # Impressão dos resultados
    print("x_sol (produção por item e período):\n", x_sol)
    print("I_sol (estoque por item e período):\n", I_sol)
    print("Custo total: ", obj)
