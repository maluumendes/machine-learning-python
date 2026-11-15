import pandas as pd

COLUNAS_X = ["studytime", "failures", "schoolsup"]

def preparar_X(df):
    """Seleciona as colunas de X e converte schoolsup para número."""
    X = df[COLUNAS_X].copy()
    # O modelo só trabalha com números: yes = 1 (tem apoio), no = 0 (não tem)
    X["schoolsup"] = X["schoolsup"].map({"yes": 1, "no": 0})
    return X


def fazer_previsao(modelo, studytime, failures, schoolsup):
    """Recebe os valores do formulário HTML, valida e retorna a previsão."""
    # Monta uma tabela com UM aluno, com as mesmas colunas usadas no treino
    novo_aluno = pd.DataFrame({
        "studytime": [int(studytime)],
        "failures": [int(failures)],
        "schoolsup": [schoolsup],
    })
    X_novo = preparar_X(novo_aluno)

    resultado = modelo.predict(X_novo)

    confianca = round(modelo.predict_proba(X_novo)[0].max() * 100, 2)

    if resultado[0] == 1:
        return {"aprovado": True, "texto": "Aprovado (nota final G3 ≥ 10)", "confianca": confianca}
    else:
        return {"aprovado": False, "texto": "Reprovado (nota final G3 < 10)", "confianca": confianca}
