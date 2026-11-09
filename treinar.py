from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_DADOS = PASTA_PROJETO / "resultados_dados" / "dados_modelo.csv"
PASTA_MODELOS = PASTA_PROJETO / "models"
PASTA_MODELOS.mkdir(exist_ok=True)

def treinar_e_avaliar():
    if not ARQUIVO_DADOS.exists():
        raise FileNotFoundError(
            f"Arquivo {ARQUIVO_DADOS} não encontrado. Execute o dados.py primeiro!"
        )

    df = pd.read_csv(ARQUIVO_DADOS, sep=";")
    
    X = df[["studytime", "failures", "schoolsup_encoded"]] if "schoolsup_encoded" in df.columns else None
    
    if X is None:
        X = df[["studytime", "failures"]].copy()
        X["schoolsup"] = df["schoolsup"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
    
    y = df["aprovado"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("=== TREINAMENTO E COMPARAÇÃO DE MODELOS ===")

    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)
    print(f"\n[Decision Tree] Acurácia: {dt_acc:.4f}")
    print("Matriz de Confusão (Decision Tree):\n", confusion_matrix(y_test, dt_pred))

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"\n[Random Forest] Acurácia: {rf_acc:.4f}")
    print("Matriz de Confusão (Random Forest):\n", confusion_matrix(y_test, rf_pred))

    melhor_modelo = rf_model if rf_acc >= dt_acc else dt_model
    nome_escolhido = "Random Forest" if rf_acc >= dt_acc else "Decision Tree"
    print(f"\n--> Modelo escolhido para a aplicação: {nome_escolhido}")

    caminho_modelo = PASTA_MODELOS / "modelo_aprovacao.pkl"
    joblib.dump(melhor_modelo, caminho_modelo)
    print(f"Modelo salvo com sucesso em: {caminho_modelo}")

if __name__ == "__main__":
    treinar_e_avaliar()