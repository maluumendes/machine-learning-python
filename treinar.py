from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from dados import carregar_dados, tratar_dados
from modelo import preparar_X

def treinar_modelos():
    # 1. Lê e trata a base padrão (student-mat.csv) usando o dados.py
    df = tratar_dados(carregar_dados())

    # 2. X = informações | y = resposta (1 = aprovado, 0 = reprovado)
    X = preparar_X(df)
    y = df["aprovado"]

    # 3. Divide ANTES de treinar: 80% treino ("estudar"), 20% teste ("prova")
    #    random_state=42 repete o mesmo sorteio; stratify mantém a proporção
    #    de aprovados e reprovados igual nos dois grupos.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # 4. Modelo 1: Árvore de Decisão
    #    max_depth=4 limita o tamanho da árvore para reduzir overfitting
    arvore = DecisionTreeClassifier(max_depth=4, random_state=42)
    arvore.fit(X_train, y_train)            # aprende somente com o treino
    pred_arvore = arvore.predict(X_test)
    acc_arvore = accuracy_score(y_test, pred_arvore)
    matriz_arvore = confusion_matrix(y_test, pred_arvore, labels=[0, 1])

    # 5. Modelo 2: Random Forest (várias árvores que "votam" na resposta)
    floresta = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    floresta.fit(X_train, y_train)
    pred_floresta = floresta.predict(X_test)
    acc_floresta = accuracy_score(y_test, pred_floresta)
    matriz_floresta = confusion_matrix(y_test, pred_floresta, labels=[0, 1])

    # 6. Referência: acurácia de quem "chuta" sempre a classe mais frequente
    acc_chute = y_test.value_counts(normalize=True).max()

    # 7. Escolha final: maior acurácia no teste.
    #    Em caso de empate, fica a Árvore de Decisão, que é mais simples de explicar.
    if acc_floresta > acc_arvore:
        nome_escolhido = "Random Forest"
        modelo_escolhido = floresta
    else:
        nome_escolhido = "Árvore de Decisão"
        modelo_escolhido = arvore

    print("\n========== TREINAMENTO E COMPARAÇÃO DE MODELOS ==========")
    print(f"Treino: {len(X_train)} alunos | Teste: {len(X_test)} alunos")
    print(f"[Árvore de Decisão] Acurácia: {acc_arvore:.4f}")
    print("Matriz de confusão (linhas = real, colunas = previsto):\n", matriz_arvore)
    print(f"[Random Forest] Acurácia: {acc_floresta:.4f}")
    print("Matriz de confusão (linhas = real, colunas = previsto):\n", matriz_floresta)
    print(f"Chute (sempre a classe mais frequente): {acc_chute:.4f}")
    print(f"--> Modelo escolhido: {nome_escolhido}")

    return {
        "modelo": modelo_escolhido,
        "nome_escolhido": nome_escolhido,
        "total_base": len(df),
        "total_treino": len(X_train),
        "total_teste": len(X_test),
        "teste_aprovados": int((y_test == 1).sum()),
        "teste_reprovados": int((y_test == 0).sum()),
        "acc_chute": round(acc_chute * 100, 2),
        "resultados": [
            {"nome": "Árvore de Decisão", "acuracia": round(acc_arvore * 100, 2),
             "matriz": matriz_arvore.tolist()},
            {"nome": "Random Forest", "acuracia": round(acc_floresta * 100, 2),
             "matriz": matriz_floresta.tolist()},
        ],
    }

if __name__ == "__main__":
    treinar_modelos()