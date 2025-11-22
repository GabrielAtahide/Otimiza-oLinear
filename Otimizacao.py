import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, GLPK_CMD

# --- Leitura do arquivo ---
def ler_arquivo(problema):
    with open(problema, 'r') as file:
        lines = file.readlines()
        m, n = map(int, lines[0].strip().split()) # le a primeira linha do arquivo e transforma em número inteiro além de retirar os espaços em branco e separa em dois
        #valores distintos 3 e 4 e não 34
        aij = np.array([list(map(float, lines[i + 1].strip().split())) for i in range(m)]) # pega a linha i+1 ou seja a segunda linha do arquivo e transforma em float 
        # os numeros alem de retirar os espaços em branco e separa em n colunas pega esses valores e coloca em uma lista e depois transforma essa
        #  lista em um array numpy (matriz )
        lj = np.array(list(map(float, lines[m + 1].strip().split())))
        dj = np.array(list(map(float, lines[m + 2].strip().split())))
        vj = np.array(list(map(float, lines[m + 3].strip().split())))
    print("Dados lidos com sucesso.")
    print(f"m={m}, n={n}")
    print(f"aij:\n{aij}")
    print(f"lj={lj}")
    print(f"dj={dj}")
    print(f"vj={vj}")
    return m, n, aij, lj, dj, vj

# --- Otimização ---
def otimizar_mix(m, n, aij, lj, dj, vj):
    # Criar modelo
    prob = LpProblem("Mix_de_Producao", LpMaximize)

    # Variáveis de decisão xj >= 0
    x = [LpVariable(f"x_{j}", lowBound=dj[j], upBound=vj[j]) for j in range(n)]

    # Função objetivo: maximizar lucro
    prob += lpSum([lj[j] * x[j] for j in range(n)]), "LucroTotal"

    # Restrições de recursos
    for i in range(m):
        prob += lpSum([aij[i, j] * x[j] for j in range(n)]) <= aij[i].sum(), f"Recurso_{i}"

    # Resolver com GLPK
    prob.solve(GLPK_CMD(msg=1))

    # Coletar resultados
    resultado = [var.varValue for var in x]
    lucro_total = sum([lj[j]*resultado[j] for j in range(n)])

    print("\n=== Resultado da Otimização ===")
    print(f"x = {resultado}")
    print(f"Lucro total = {lucro_total}")

    return resultado, lucro_total

# --- Salvar resultado ---
def salvar_arquivo(nome, x, lucro):
    with open(nome, 'w') as f:
        f.write("Solução do Mix de Produção\n")
        f.write("x_j:\n")
        f.write(" ".join(map(str, x)) + "\n")
        f.write(f"Lucro total: {lucro}\n")
    print(f"Resultados salvos em {nome}")

# --- Execução ---
m, n, aij, lj, dj, vj = ler_arquivo("problema.txt")
x, lucro = otimizar_mix(m, n, aij, lj, dj, vj)
salvar_arquivo("output.txt", x, lucro)
