from pathlib import Path
import joblib
import pandas as pd

PASTA_PROJETO = Path(__file__).resolve().parent
CAMINHO_MODELO = PASTA_PROJETO / "models" / "modelo_aprovacao.pkl"

def carregar_modelo():
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            "Modelo treinado não encontrado. Execute o treinar.py primeiro!"
        )
    return joblib.load(CAMINHO_MODELO)

def fazer_previsao(studytime, failures, schoolsup):
    """Recebe os valores do formulário HTML, valida e retorna a previsão."""
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
