from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")  #gera imagens sem abrir janela (necessário no servidor)
import matplotlib.pyplot as plt

from flask import Flask, render_template, request

from dados import carregar_dados, tratar_dados, gerar_analises, COLUNAS_NECESSARIAS
from modelo import fazer_previsao, carregar_modelo, CAMINHO_MODELO


app = Flask(__name__)
#limite de tamanho do arquivo enviado = 2 MB
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

PASTA_PROJETO = Path(__file__).resolve().parent
PASTA_STATIC = PASTA_PROJETO / "static"
PASTA_UPLOADS = PASTA_PROJETO / "uploads"
PASTA_STATIC.mkdir(exist_ok=True)
PASTA_UPLOADS.mkdir(exist_ok=True)

#valores aceitos no formulário de previsão (mesma codificação da base UCI)
VALORES_ACEITOS = {
    "studytime": ["1", "2", "3", "4"],
    "failures": ["0", "1", "2", "3"],
    "schoolsup": ["yes", "no"],
}

#informacos sobre o modelo fixo usado na previsao
BASE_TREINO = "student-mat.csv (UCI Student Performance - Matemática)"
MODELOS = {
    "DecisionTreeClassifier": "Árvore de Decisão (DecisionTreeClassifier)",
    "RandomForestClassifier": "Random Forest (RandomForestClassifier)",
}


#LEITURA E VALIDACAO DO CSV ENVIADO -------------------------------------------------

def ler_csv_enviado(caminho):

    if caminho.stat().st_size == 0:
        raise ValueError("O arquivo enviado está vazio.")

    try:
        # A base original usa ponto e vírgula. Tentamos ";" primeiro.
        separador = ";"
        df = pd.read_csv(caminho, sep=separador)

        # Se veio só uma coluna, provavelmente o separador é vírgula.
        if len(df.columns) == 1:
            separador = ","
            df = pd.read_csv(caminho, sep=separador)
    except pd.errors.EmptyDataError:
        raise ValueError("O arquivo enviado está vazio.")
    except UnicodeDecodeError:
        raise ValueError("Não foi possível ler o arquivo. Salve o CSV com codificação UTF-8.")
    except pd.errors.ParserError:
        raise ValueError("O arquivo não está em um formato CSV válido.")

    if len(df.columns) == 1:
        raise ValueError("Separador não reconhecido. Use ponto e vírgula (;) ou vírgula (,).")

    df.columns = df.columns.astype(str).str.strip()

    colunas_ausentes = [col for col in COLUNAS_NECESSARIAS if col not in df.columns]
    if colunas_ausentes:
        raise ValueError(
            "O arquivo não possui as colunas obrigatórias: " + ", ".join(colunas_ausentes)
        )

    if df.empty:
        raise ValueError("O arquivo possui apenas o cabeçalho, sem registros.")

    return df, separador


#GRAFICOS ----------------------------------------------------------------------------
def gerar_graficos(df):
    df = df.copy()
    df["studytime"] = df["studytime"].astype(int)
    df["failures"] = df["failures"].astype(int)

    # Gráfico 1 - Distribuição: quantidade de aprovados e reprovados
    distribuicao = df["aprovado"].value_counts().reindex([0, 1], fill_value=0)
    distribuicao.index = ["Reprovados", "Aprovados"]
    plt.figure(figsize=(6, 4))
    distribuicao.plot(kind="bar", color=["#F4A6C1", "#A8E6CF"])
    plt.title("Distribuição de Alunos por Situação Final")
    plt.xlabel("Situação (aprovado = G3 ≥ 10)")
    plt.ylabel("Quantidade de alunos")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PASTA_STATIC / "grafico_distribuicao.png", dpi=150)
    plt.close()

    # Gráfico 2 - Comparação: taxa de aprovação por tempo de estudo
    taxa_estudo = df.groupby("studytime")["aprovado"].mean() * 100
    plt.figure(figsize=(6, 4))
    taxa_estudo.plot(kind="bar", color=['#F4A6C1', '#CDB4DB', '#FDE8A8', '#A8E6CF'])
    plt.title("Taxa de Aprovação por Tempo de Estudo Semanal")
    plt.xlabel("Tempo de estudo (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
    plt.ylabel("Aprovação (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PASTA_STATIC / "grafico_comparacao.png", dpi=150)
    plt.close()

    # Gráfico 3 - Relação: nota final (G3) por reprovações anteriores
    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column="G3", by="failures", ax=ax, grid=False)
    ax.axhline(10, color="#B5527A", linestyle="--", label="Nota mínima (10)")
    plt.suptitle("")  # remove o título automático do boxplot do Pandas
    plt.title("Nota Final (G3) por Reprovações Anteriores")
    plt.xlabel("Número de reprovações anteriores (failures)")
    plt.ylabel("Nota final G3 (0 a 20)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PASTA_STATIC / "grafico_dispersao.png", dpi=150)
    plt.close()


#FUNCAO COMUM PARA MONTAR A PAGINA DE ANALISE --------------------------------------

def exibir_analise(df_original, nome_arquivo, separador):
    df_tratado = tratar_dados(df_original)

    if df_tratado.empty:
        raise ValueError(
            "Nenhum registro válido após o tratamento. Verifique os valores das colunas "
            "studytime (1-4), failures (0-3), schoolsup (yes/no) e G3 (0-20)."
        )

    gerar_graficos(df_tratado)
    analises = gerar_analises(df_tratado)

    tabelas_analise = {
        "Aprovação por tempo de estudo (studytime)": analises["por_tempo_estudo"],
        "Aprovação por reprovações anteriores (failures)": analises["por_reprovacoes"],
        "Aprovação por apoio escolar (schoolsup)": analises["por_apoio"],
    }
    tabelas_html = {
        titulo: tabela.to_html(classes="table table-sm table-bordered", index=False)
        for titulo, tabela in tabelas_analise.items()
    }

    return render_template(
        "analise.html",
        nome_arquivo=nome_arquivo,
        separador=separador,
        total_original=len(df_original),
        total_colunas=len(df_original.columns),
        total=len(df_tratado),
        removidos=len(df_original) - len(df_tratado),
        taxa=round(df_tratado["aprovado"].mean() * 100, 2),
        tabela=df_tratado.head(10).to_html(
            classes="table table-striped table-bordered", index=False
        ),
        tabelas_analise=tabelas_html,
        # Muda a cada análise para o navegador não mostrar gráficos antigos (cache)
        versao_graficos=datetime.now().strftime("%Y%m%d%H%M%S"),
    )


#INFORMACOES DO MODELO ----------------------------------------------------------
def obter_info_modelo():
    modelo = carregar_modelo()
    nome_tecnico = type(modelo).__name__
    data_treino = datetime.fromtimestamp(CAMINHO_MODELO.stat().st_mtime)
    return {
        "nome": MODELOS.get(nome_tecnico, nome_tecnico),
        "base": BASE_TREINO,
        "data": data_treino.strftime("%d/%m/%Y %H:%M"),
        "variaveis": "studytime, failures, schoolsup",
    }


#ROTAS  ------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", colunas=COLUNAS_NECESSARIAS)


@app.route("/upload", methods=["POST"])
def upload_csv():
    # O formulário de upload precisa de POST: arquivos não são enviados por GET.
    arquivo = request.files.get("arquivo_csv")

    if arquivo is None or arquivo.filename == "":
        return render_template("index.html", colunas=COLUNAS_NECESSARIAS,
                               erro="Nenhum arquivo selecionado.")

    if not arquivo.filename.lower().endswith(".csv"):
        return render_template("index.html", colunas=COLUNAS_NECESSARIAS,
                               erro="Formato inválido. Envie um arquivo com extensão .csv.")

    # Guarda o arquivo original sem alterações (o tratamento é feito em uma cópia)
    nome_salvo = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".csv"
    caminho = PASTA_UPLOADS / nome_salvo
    arquivo.save(caminho)

    try:
        df_original, separador = ler_csv_enviado(caminho)
        return exibir_analise(df_original, arquivo.filename, separador)
    except ValueError as erro:
        return render_template("index.html", colunas=COLUNAS_NECESSARIAS, erro=str(erro))
    except Exception as erro:
        print("Erro inesperado no upload:", erro)  # detalhe técnico fica no terminal
        return render_template("index.html", colunas=COLUNAS_NECESSARIAS,
                               erro="Não foi possível processar o arquivo. Verifique o conteúdo do CSV.")


@app.route("/analise-padrao")
def analise_padrao():
    try:
        df_original = carregar_dados()  # student-mat.csv (separador ;)
        return exibir_analise(df_original, "student-mat.csv (base padrão)", ";")
    except Exception as erro:
        print("Erro na base padrão:", erro)
        return render_template("index.html", colunas=COLUNAS_NECESSARIAS,
                               erro="Não foi possível carregar a base padrão.")


@app.route("/previsao", methods=["GET", "POST"])
def previsao():
    resultado = None
    erro = None

    try:
        info_modelo = obter_info_modelo()
    except FileNotFoundError:
        return render_template("previsao.html", resultado=None, info_modelo=None,
                               erro="Modelo não encontrado. Execute python treinar.py antes de usar a previsão.")

    if request.method == "POST":
        studytime = request.form.get("studytime", "").strip()
        failures = request.form.get("failures", "").strip()
        schoolsup = request.form.get("schoolsup", "").strip().lower()

        # Validação: cada campo precisa ter um dos valores aceitos
        if studytime not in VALORES_ACEITOS["studytime"]:
            erro = "Tempo de estudo inválido. Escolha uma opção de 1 a 4."
        elif failures not in VALORES_ACEITOS["failures"]:
            erro = "Número de reprovações inválido. Escolha uma opção de 0 a 3."
        elif schoolsup not in VALORES_ACEITOS["schoolsup"]:
            erro = "Apoio escolar inválido. Escolha Sim ou Não."
        else:
            resultado = fazer_previsao(studytime, failures, schoolsup)

    return render_template("previsao.html", resultado=resultado,
                           erro=erro, info_modelo=info_modelo)


@app.errorhandler(413)
def arquivo_grande(erro):
    return render_template("index.html", colunas=COLUNAS_NECESSARIAS,
                           erro="Arquivo muito grande. O limite é de 2 MB."), 413


if __name__ == "__main__":
    app.run(debug=True)