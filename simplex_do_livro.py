import numpy as np

EPS = 1e-9

def atualiza_inversa(B_inv, y, idx_saida):
    """
    Atualiza a inversa da base (B_inv) de forma eficiente usando a matriz elementar E.
    y: vetor = B_inv @ a_entrada  (mostra o efeito da variável que entra)
    idx_saida: índice (0..m-1) da variável que sai da base
    Retorna: nova inversa B'_inv = E @ B_inv
    """
    m = B_inv.shape[0]                     # número de variáveis básicas (linhas de B_inv)
    e_l = np.zeros(m)                      # vetor coluna de zeros
    e_l[idx_saida] = 1.0                   # coloca 1 na posição da variável que sai
    y_l = y[idx_saida]                     # elemento correspondente à linha de saída

    if abs(y_l) < EPS:
        raise ZeroDivisionError("y_l muito próximo de zero ao atualizar inversa")

    E = np.eye(m)                          # matriz identidade m x m
    E[:, idx_saida] = (y - e_l) / y_l      # substitui a coluna de saída pela fórmula elementar

    return E @ B_inv                       # retorna a nova inversa da base


def simplex_core(A, b, c, base_indices, max_iters=1000, fase=2):
    """
    Núcleo principal do método Simplex.
    Executa iterações até encontrar a solução ótima, ou identificar se o problema é ilimitado.

    Parâmetros:
      A, b, c: matriz, vetor de restrições e vetor de custos
      base_indices: índices das colunas que formam a base inicial
      fase: indica se é fase 1 ou 2 (apenas para exibir mensagens)

    Retorna:
      Dicionário com status, solução, base final, inversa da base e número de iterações
    """
    m, n = A.shape # cria dimensões da matriz A com m restrições e n variáveis
    base = list(base_indices) #guarda os índices da base inicial em uma lista
    nao_base = [j for j in range(n) if j not in base] #cria uma lista complementar com os índices das variáveis fora da base

    B = A[:, base] #guarda em B a coluna da matriz A correspondente às variáveis básicas(base)
    B_inv = np.linalg.inv(B) #calcula a inversa da matriz B
    it = 0

    while it < max_iters:
        it += 1
        x_B = B_inv @ b                      # solução básica atual
        c_B = c[base]                        # custos das variáveis básicas
        lambd = c_B @ B_inv                  # multiplicadores simplex (λ)
        
        # custos reduzidos das variáveis não básicas
        reduzido = {j: c[j] - lambd @ A[:, j] for j in nao_base} #calcula o custo reduzido para cada variável fora da base

        # verifica se todos custos reduzidos >= 0 (condição de otimalidade)
        j_entra = None
        menor_custo = 0.0
        for j in sorted(nao_base): # O sorted garante que, em caso de empate, a variável com menor índice entre as candidatas seja escolhida, escolhendo a variável que entra na base
            if reduzido[j] < menor_custo - EPS: #verifica se o custo reduzido é negativo (indicando que a variável pode entrar na base)
                menor_custo = reduzido[j] #atualiza o menor custo reduzido encontrado
                j_entra = j #atualiza o índice da variável que entra na base

        if j_entra is None:
            # ótimo encontrado
            x = np.zeros(n)
            for idx, bi in enumerate(base):
                x[bi] = max(0.0, x_B[idx])   # garante não-negatividade
            return {
                "status": "ótimo",
                "x": x,
                "valor": float(c @ x),
                "base": base,
                "B_inv": B_inv,
                "lambda": lambd,
                "iterações": it
            }

        # cálculo da direção simplex
        y = B_inv @ A[:, j_entra]
        if np.all(y <= EPS): #teste de ilimitabilidade raio direção
            return {"status": "ilimitado", "variável_entrando": j_entra, "iterações": it}

        # teste da razão mínima (para determinar quem sai da base)
        razoes = np.full(m, np.inf) #cria um vetor de razões inicializado com infinito
        for i in range(m): #para cada variável básica (calcula a razão entre a solução básica e o vetor y) onde y é a direção simplex
            if y[i] > EPS: #verifica se o elemento y[i] é positivo (indicando que a variável pode ser reduzida)
                razoes[i] = x_B[i] / y[i] #calcula a razão para a variável básica i
        min_ratio = razoes.min()   #encontra a menor razão positiva
        idx_saida = int(np.where(np.isclose(razoes, min_ratio))[0][0]) # guarda em idx_saida o índice da variável que sai da base quando comparada com a menor razão

        # atualiza a inversa da base
        B_inv = atualiza_inversa(B_inv, y, idx_saida)

        # troca variáveis da base e fora da base
        var_sai = base[idx_saida]
        base[idx_saida] = j_entra
        nao_base.remove(j_entra)
        nao_base.append(var_sai)
        nao_base.sort()

    return {"status": "limite de iterações atingido", "iterações": it}


def fase1(A, b):
    """
    Constrói o problema auxiliar da Fase 1 do Simplex.
    Adiciona variáveis artificiais para encontrar uma base viável inicial.

    Retorna:
      (A1, b1, c1, base_inicial, n_original)
    """
    m, n = A.shape
    A1 = np.hstack([A.copy(), np.eye(m)])  # adiciona variáveis artificiais
    c1 = np.hstack([np.zeros(n), np.ones(m)])  # minimiza soma das artificiais
    base_inicial = list(range(n, n + m))
    return A1, b.copy(), c1, base_inicial, n


def remove_artificiais_fase1(A, b, c, B_inv, base, n_orig):
    """
    Remove variáveis artificiais da base após a Fase 1 (quando o valor ótimo é 0).
    Substitui cada variável artificial por uma variável original viável.
    """
    m = A.shape[0]
    tentativas = 0

    while any(bi >= n_orig for bi in base) and tentativas < m * 2: # limite de tentativas sendo 2*m
        tentativas += 1
        for linha in range(m):
            bi = base[linha]
            if bi >= n_orig:
                for j in range(n_orig):
                    if j in base:
                        continue
                    a_j = A[:, j]
                    y = B_inv @ a_j
                    if abs(y[linha]) > EPS:
                        B_inv = atualiza_inversa(B_inv, y, linha)
                        base[linha] = j
                        break
                else:
                    raise RuntimeError(
                        "Não foi possível remover variável artificial da base (possível redundância)."
                    )
    return B_inv, base


def simplex_com_fase1(A, b, c, base_inicial=None, max_iters=1000):
    """
    Implementa o Método Simplex completo com Fase 1 + Fase 2.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    m, n = A.shape

    # garante que b >= 0
    for i in range(m):
        if b[i] < -EPS:
            A[i, :] *= -1
            b[i] *= -1

    total_iters = 0

    if base_inicial is None:
        # executa a fase 1
        A1, b1, c1, base1, n_orig = fase1(A, b)
        res1 = simplex_core(A1, b1, c1, base1, max_iters, fase=1)

        if res1["status"] != "ótimo":
            return {"status": "falha_fase1", "detalhe": res1}

        total_iters += res1["iterações"]

        if abs(res1["valor"]) > 1e-7:
            return {"status": "inviável", "valor_fase1": res1["valor"]}

        # remove variáveis artificiais
        B_inv = res1["B_inv"]
        base = res1["base"][:]

        try:
            B_inv, base = remove_artificiais_fase1(A1, b1, c1, B_inv, base, n_orig)
        except RuntimeError as e:
            return {"status": "falha_remover_artificiais", "erro": str(e)}

        base = [int(bi) for bi in base]
        res2 = simplex_core(A, b, c, base, max_iters, fase=2)
        res2["iterações_totais"] = total_iters + res2["iterações"]
        return res2

    else:
        # base inicial já viável fornecida
        return simplex_core(A, b, c, base_inicial, max_iters, fase=2)


# Exemplo de uso:
if __name__ == "__main__":
    # Exemplo de minimização
    A = np.array([[1.0, 1.0, 1.0, 0.0],
                  [2.0, 0.5, 0.0, 1.0]])
    b = np.array([5.0, 8.0])
    c = np.array([-3.0, -2.0, 0.0, 0.0])  # minimiza c^T x

    resultado = simplex_com_fase1(A, b, c)
    print("Status:", resultado["status"])

    if resultado["status"] == "ótimo":
        print("Solução ótima:", resultado["x"])
        print("Valor ótimo:", resultado["valor"])
        print("Base final:", resultado["base"])
        print("Iterações:", resultado["iterações"])
    else:
        print(resultado)
