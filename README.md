# Sistema de Previsão de Aprovação Escolar

Trabalho Final – Tópicos Especiais em Ciência da Computação
Aplicação Web com Flask, análise de dados com Pandas e Machine Learning com scikit-learn.

## Tema e pergunta

**Tema 01 – Aprovação de estudantes (classificação).**
Pergunta: *com o tempo de estudo, as reprovações anteriores e o apoio escolar de um aluno, é possível prever se ele será aprovado em Matemática?*

## Dataset

- **Fonte:** UCI Machine Learning Repository – Student Performance (arquivo `student-mat.csv`, disciplina de Matemática).
  https://archive.ics.uci.edu/dataset/320/student+performance
- **Registros:** 395 alunos | **Separador:** ponto e vírgula (`;`)

### Dicionário das colunas usadas

| Coluna | Significado | Valores aceitos |
|---|---|---|
| `studytime` | Tempo de estudo semanal | 1 (< 2h), 2 (2 a 5h), 3 (5 a 10h), 4 (> 10h) |
| `failures` | Reprovações anteriores | 0, 1, 2, 3 (3 = três ou mais) |
| `schoolsup` | Apoio educacional extra | `yes` ou `no` |
| `G3` | Nota final (0 a 20) | 0 a 20 — usada só para criar o alvo |

## Instalação e execução

```bash
pip install -r requirements.txt
python app.py
```

Abra no navegador: http://127.0.0.1:5000/

Ao iniciar, o `app.py` treina e compara os dois modelos (os resultados aparecem no terminal e na página de previsão). Não é preciso rodar nenhum outro script antes.

Scripts opcionais, para ver os resultados no terminal:

- `python dados.py` – diagnóstico, tratamento e análises; salva tabelas na pasta `resultados_dados/`.
- `python treinar.py` – treino, teste e comparação dos modelos.

## Esquema do CSV aceito no upload

- Extensão `.csv`, separador `;` ou `,`, texto em UTF-8.
- Precisa conter as colunas `studytime`, `failures`, `schoolsup` e `G3` (outras colunas são ignoradas).
- Validações: arquivo vazio, extensão, separador, colunas obrigatórias, arquivo só com cabeçalho e valores fora das faixas da tabela acima.
- Linhas com valores ausentes ou inválidos são removidas, e a página mostra quantas foram removidas.
- O arquivo original é guardado sem alterações na pasta `uploads/`.

## Tratamento dos dados (dados.py)

- Conversão de `studytime`, `failures` e `G3` para número; valores que não são números viram ausentes.
- Padronização de `schoolsup` (espaços removidos, letras minúsculas).
- Remoção das linhas com valores ausentes ou fora das faixas aceitas.
- Duplicatas são contadas, mas não removidas automaticamente.
- Criação do alvo: `aprovado = 1` se `G3 ≥ 10`, senão `0`.
- Na base padrão, nenhum registro foi removido (395 antes e depois).

## Principais achados

1. **Classes desbalanceadas:** 265 aprovados (67%) e 130 reprovados (33%).
2. **Reprovações anteriores pesam mais:** a mediana de G3 cai de 11 (nenhuma reprovação) para 7 (três reprovações).
3. **Tempo de estudo ajuda pouco:** a aprovação vai de cerca de 65% (até 5h por semana) para cerca de 75% (5h ou mais).

## Gráficos

1. **Distribuição:** quantidade de aprovados e reprovados.
2. **Comparação:** taxa de aprovação por tempo de estudo.
3. **Relação:** boxplot da nota final (G3) por reprovações anteriores.

## Machine Learning

- **X:** `studytime`, `failures`, `schoolsup` (`yes` = 1, `no` = 0).
- **y:** `aprovado` (G3 ≥ 10).
- **G3 não entra em X**, porque é ele que define a resposta. Usá-lo seria entregar o gabarito ao modelo.
- **Divisão:** `train_test_split` com `test_size=0.20`, `random_state=42` e `stratify=y` → 316 alunos para treino e 79 para teste (53 aprovados e 26 reprovados).
- **Os dois modelos usam a mesma divisão.**

### Comparação dos modelos

| Modelo | Acurácia no teste | Reprovados identificados | Reprovados previstos como aprovados (FP) |
|---|---|---|---|
| Árvore de Decisão (`max_depth=4`) | **68,35%** | 9 de 26 | 17 |
| Random Forest (`n_estimators=100`, `max_depth=5`) | 65,82% | 7 de 26 | 19 |
| Chute (sempre "aprovado") | 67,09% | 0 de 26 | 26 |

### Escolha final

A **Árvore de Decisão** foi escolhida porque:

- teve a maior acurácia no teste;
- identificou mais alunos que realmente reprovaram;
- é o modelo mais simples de explicar.

### Limitações

- A acurácia ficou só um pouco acima do chute (68,35% contra 67,09%). Três variáveis explicam pouco da nota final.
- O modelo erra principalmente prevendo "aprovado" para alunos que reprovaram (falso positivo). Num uso real, esse é o erro mais grave, porque o aluno em risco não seria identificado.
- O modelo é treinado sempre com a base padrão. O CSV enviado no upload atualiza análises e gráficos, mas não o modelo.
- Base pequena (395 alunos) e de duas escolas portuguesas, então não representa qualquer escola.

## Estrutura do projeto

```
app.py              rotas Flask, upload, gráficos e integração
dados.py            leitura, diagnóstico, tratamento e análises (Pandas)
modelo.py           preparação do X e função de previsão (predict)
treinar.py          treino, teste e comparação dos modelos
student-mat.csv     base de dados padrão
templates/          index.html, analise.html, previsao.html
static/             CSS e imagens dos gráficos
resultados_dados/   tabelas geradas pelo dados.py
```

## Integrantes e responsabilidades

| Integrante | Responsabilidade | Arquivos |
|---|---|---|
| Sarah de Moura Silva | Dados: leitura, limpeza e análise | `dados.py` |
| Maria Luiza Cordeiro Lisboa | Machine Learning: modelos, treino e avaliação | `modelo.py`, `treinar.py` |
| Julia Soares Catharino | Front-end: páginas HTML e gráficos | `templates/`, `static/` |
| Maria Luíza Barcelos Mendes | Back-end e integração: Flask, rotas, upload e validação | `app.py` |

## Fontes e ferramentas

- Dataset: Cortez, P. & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. UCI Machine Learning Repository.
- Bibliotecas: Flask, Pandas, Matplotlib, scikit-learn; Bootstrap 5 (CSS).
- Material das aulas da disciplina.
- Ferramenta de IA utilizada como apoio na revisão.
