from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_DADOS = PASTA_PROJETO / "resultados_dados" / "dados_modelo.csv"

# Truque de compatibilidade: aponta para o CSV de dados para o app.py 
# conseguir ler a data de modificação sem precisar do arquivo .pkl
CAMINHO_MODELO = ARQUIVO_DADOS

# Variável global para guardar o modelo treinado na memória RAM
_modelo_em_memoria = None

def carregar_modelo():
    """Treina o modelo na memória RAM na primeira vez e o reutiliza nas próximas."""
    global _modelo_em_memoria
    
    if _modelo_em_memoria is not None:
        return _modelo_em_memoria

    if not ARQUIVO_DADOS.exists():
        raise FileNotFoundError(
            f"Arquivo {ARQUIVO_DADOS} não encontrado. Execute o dados.py primeiro!"
        )

    df = pd.read_csv(ARQUIVO_DADOS, sep=";")
    
    # Prepara as variáveis de entrada (X) e o alvo (y)
    X = df[["studytime", "failures"]].copy()
    X["schoolsup"] = df["schoolsup"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
    y = df["aprovado"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Treina o modelo Random Forest diretamente na memória
    modelo = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    modelo.fit(X_train, y_train)

    _modelo_em_memoria = modelo
    return _modelo_em_memoria

def fazer_previsao(studytime, failures, schoolsup):
    """Recebe os valores do formulário HTML, obtém o modelo da memória e retorna a previsão."""
    modelo = carregar_modelo()

    schoolsup_num = 1 if str(schoolsup).strip().lower() in ["yes", "1", "sim", "true"] else 0

    dados_entrada = pd.DataFrame([[int(studytime), int(failures), schoolsup_num]], 
                                   columns=["studytime", "failures", "schoolsup"])
    
    predicao = modelo.predict(dados_entrada)[0]
    probabilidade = modelo.predict_proba(dados_entrada)[0].max() if hasattr(modelo, "predict_proba") else None
    
    resultado_texto = "Aprovado (G3 ≥ 10)" if predicao == 1 else "Reprovado (G3 < 10)"
    return {
        "predicao_int": int(predicao),
        "resultado_texto": resultado_texto,
        "confianca": round(float(probabilidade) * 100, 2) if probabilidade else None
    }
