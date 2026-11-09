from pathlib import Path
import pandas as pd


PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_CSV = PASTA_PROJETO / "student-mat.csv"
PASTA_RESULTADOS = PASTA_PROJETO / "resultados_dados"

# Recorte definido no enunciado do Tema 01.
COLUNAS_ENTRADA = ["studytime", "failures", "schoolsup"]
COLUNA_NOTA_FINAL = "G3"
COLUNAS_NECESSARIAS = COLUNAS_ENTRADA + [COLUNA_NOTA_FINAL]


# ------------------------------------------------------------
# 1. LEITURA DO CSV
# ------------------------------------------------------------
def carregar_dados(caminho=ARQUIVO_CSV):
    """Lê o CSV de Matemática, que utiliza ponto e vírgula como separador."""
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}\n"
            "Coloque o arquivo student-mat.csv na mesma pasta de dados.py."
        )

    df = pd.read_csv(caminho, sep=";")
    df.columns = df.columns.astype(str).str.strip()

    colunas_ausentes = [col for col in COLUNAS_NECESSARIAS if col not in df.columns]
    if colunas_ausentes:
        raise ValueError(
            "O CSV não contém todas as colunas necessárias. "
            f"Colunas ausentes: {colunas_ausentes}"
        )

    return df


# ------------------------------------------------------------
# 2. DIAGNÓSTICO DA BASE ORIGINAL
# ------------------------------------------------------------
def diagnosticar_dados(df):
    """Mostra estrutura, valores ausentes, duplicatas e resumo das colunas usadas."""
    print("\n========== DIAGNÓSTICO DA BASE ORIGINAL ==========")
    print(f"Quantidade de registros: {len(df)}")
    print(f"Quantidade de colunas: {len(df.columns)}")

    print("\nPrimeiras cinco linhas:")
    print(df.head())

    print("\nTipos de dados e quantidade de valores preenchidos:")
    df.info()

    print("\nValores ausentes por coluna:")
    ausentes = df.isna().sum()
    print(ausentes[ausentes > 0] if ausentes.sum() else "Nenhum valor ausente encontrado.")

    quantidade_duplicadas = int(df.duplicated().sum())
    print(f"\nLinhas totalmente duplicadas: {quantidade_duplicadas}")
    if quantidade_duplicadas:
        print(
            "Atenção: as duplicatas foram identificadas, mas não serão apagadas "
            "automaticamente. O grupo deve verificar se são registros repetidos "
            "ou observações diferentes com os mesmos valores."
        )

    print("\nResumo das colunas utilizadas no projeto:")
    print(df[COLUNAS_NECESSARIAS].describe(include="all"))


# ------------------------------------------------------------
# 3. TRATAMENTO E VALIDAÇÃO
# ------------------------------------------------------------
def tratar_dados(df):
    """Padroniza e valida as quatro colunas necessárias ao recorte do projeto.

    Linhas com informações ausentes ou valores incompatíveis nessas colunas são
    separadas da base tratada. Duplicatas não são removidas automaticamente.
    """
    dados = df.copy()

    # Converte os campos numéricos. Valores que não puderem ser convertidos
    # tornam-se ausentes e serão identificados na validação abaixo.
    for coluna in ["studytime", "failures", "G3"]:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")

    # Padroniza texto: por exemplo, " YES " passa a ser "yes".
    dados["schoolsup"] = (
        dados["schoolsup"].astype("string").str.strip().str.lower()
    )

    # Valores aceitos conforme a codificação documentada da base.
    studytime_invalido = ~dados["studytime"].isin([1, 2, 3, 4])
    failures_invalido = ~dados["failures"].isin([0, 1, 2, 3])
    schoolsup_invalido = ~dados["schoolsup"].isin(["yes", "no"])
    g3_invalido = ~dados["G3"].between(0, 20, inclusive="both")

    verificacoes = pd.DataFrame(
        {
            "studytime": studytime_invalido,
            "failures": failures_invalido,
            "schoolsup": schoolsup_invalido,
            "G3": g3_invalido,
        },
        index=dados.index,
    )

    print("\n========== VERIFICAÇÃO DE VALORES ==========")
    print("Quantidade de valores ausentes/incompatíveis por coluna:")
    print(verificacoes.sum())

    linhas_invalidas = verificacoes.any(axis=1)
    quantidade_removida = int(linhas_invalidas.sum())
    dados_tratados = dados.loc[~linhas_invalidas].copy()

    print(f"\nRegistros antes do tratamento: {len(dados)}")
    print(f"Registros removidos por dados ausentes/incompatíveis nas colunas do projeto: {quantidade_removida}")
    print(f"Registros depois do tratamento: {len(dados_tratados)}")
    print(
        "Linhas duplicadas não foram removidas automaticamente; a quantidade "
        f"encontrada na base original foi {int(df.duplicated().sum())}."
    )

    # Cria a variável-alvo definida pelo professor: G3 >= 10 é aprovado.
    dados_tratados["aprovado"] = (dados_tratados["G3"] >= 10).astype(int)

    return dados_tratados


# ------------------------------------------------------------
# 4. ANÁLISES PARA RESPONDER À PERGUNTA DO PROJETO
# ------------------------------------------------------------
def analisar_por_grupo(df, coluna):
    """Calcula quantidade de alunos e taxa de aprovação para cada categoria."""
    resultado = (
        df.groupby(coluna)["aprovado"]
        .agg(quantidade="count", aprovados="sum", taxa_aprovacao="mean")
        .reset_index()
    )
    resultado["reprovados"] = resultado["quantidade"] - resultado["aprovados"]
    resultado["taxa_aprovacao_pct"] = (resultado["taxa_aprovacao"] * 100).round(2)
    resultado = resultado.drop(columns=["taxa_aprovacao"])
    return resultado


def gerar_analises(df):
    """Gera tabelas de análise sobre estudo, reprovações anteriores e apoio escolar."""
    print("\n========== ANÁLISES DO PROJETO ==========")

    distribuicao = (
        df["aprovado"]
        .value_counts()
        .reindex([1, 0], fill_value=0)
        .rename(index={1: "Aprovado", 0: "Reprovado"})
        .rename_axis("situacao")
        .reset_index(name="quantidade")
    )

    por_tempo_estudo = analisar_por_grupo(df, "studytime")
    por_reprovacoes = analisar_por_grupo(df, "failures")
    por_apoio = analisar_por_grupo(df, "schoolsup")

    print("\nDistribuição de aprovados e reprovados:")
    print(distribuicao.to_string(index=False))

    print("\nAprovação por tempo semanal de estudo (studytime):")
    print(por_tempo_estudo.to_string(index=False))

    print("\nAprovação por reprovações anteriores (failures):")
    print(por_reprovacoes.to_string(index=False))

    print("\nAprovação por apoio escolar extra (schoolsup):")
    print(por_apoio.to_string(index=False))

    return {
        "distribuicao": distribuicao,
        "por_tempo_estudo": por_tempo_estudo,
        "por_reprovacoes": por_reprovacoes,
        "por_apoio": por_apoio,
    }


# ------------------------------------------------------------
# 5. SALVAMENTO DOS ARQUIVOS PARA O GRUPO
# ------------------------------------------------------------
def salvar_resultados(df_tratado, analises):
    PASTA_RESULTADOS.mkdir(exist_ok=True)

    # Mantém as colunas originais e acrescenta a situação de aprovação.
    df_tratado.to_csv(
        PASTA_RESULTADOS / "dados_tratados_completo.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    # Arquivo mais enxuto para o integrante 2: contém apenas X e y.
    # G3 fica de fora para evitar que a nota final entre como variável de entrada.
    dados_modelo = df_tratado[COLUNAS_ENTRADA + ["aprovado"]].copy()
    dados_modelo.to_csv(
        PASTA_RESULTADOS / "dados_modelo.csv",
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    nomes_arquivos = {
        "distribuicao": "distribuicao_aprovacao.csv",
        "por_tempo_estudo": "aprovacao_por_tempo_estudo.csv",
        "por_reprovacoes": "aprovacao_por_reprovacoes.csv",
        "por_apoio": "aprovacao_por_apoio_escolar.csv",
    }
    for chave, nome_arquivo in nomes_arquivos.items():
        analises[chave].to_csv(
            PASTA_RESULTADOS / nome_arquivo,
            sep=";",
            index=False,
            encoding="utf-8-sig",
        )

    print(f"\nArquivos salvos na pasta: {PASTA_RESULTADOS}")


def main():
    df_original = carregar_dados()
    diagnosticar_dados(df_original)

    df_tratado = tratar_dados(df_original)
    if df_tratado.empty:
        raise ValueError(
            "Nenhum registro válido restou após a validação. "
            "Revise o CSV antes de continuar."
        )

    analises = gerar_analises(df_tratado)
    salvar_resultados(df_tratado, analises)

    print("\nProcesso concluído. Confira os arquivos da pasta resultados_dados.")


if __name__ == "__main__":
    main()
