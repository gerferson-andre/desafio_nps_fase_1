# 📊 Tech Challenge: Análise Preditiva e Diagnóstico de NPS

Solução do **Tech Challenge (Fase 1)** da Pós-Tech em AI Scientist (FIAP). A análise examina uma base de 2.500 pedidos de e-commerce para entender o que se associa à nota de NPS dos clientes e testar se essa nota pode ser estimada a partir de sinais operacionais registrados antes da pesquisa.

---

## 📌 Sumário Executivo

A base tem alto índice de insatisfação: **índice NPS de −79,96**, com 84,36% de detratores, 11,24% de neutros e 4,40% de promotores.

### Principais achados

- **Distribuição da nota:** média de 4,38 (desvio padrão de 2,51). A nota não segue distribuição normal (Shapiro-Wilk: W = 0,9817, p < 0,001), o que levou ao uso de Spearman e de testes não paramétricos.
- **Associações com a nota (Spearman):** as associações mais fortes são negativas e vêm dos sinais de atrito: atraso na entrega (ρ = −0,59), número de reclamações (ρ = −0,49), contatos com o SAC (ρ = −0,34) e tempo de resolução (ρ = −0,19). Frete, idade, parcelas, desconto e valor do pedido ficam próximos de zero.
- **Diferença entre detratores e demais clientes:** detratores têm, em média, 2,4 dias de atraso e 4,4 reclamações, contra 1,2 dia e 2,8 reclamações nos não detratores sem recompra.
- **Recompra:** quem recompra em 30 dias tem nota média de 9,0, contra 3,9 de quem não recompra (Mann-Whitney U, p < 0,001). Os grupos não se sobrepõem: a maior nota sem recompra é 7,9 e a menor com recompra é 8,0. Essa separação total exige investigar a origem da variável antes de ser usada como evidência.
- **Região:** não há diferença de nota entre regiões (Kruskal-Wallis, H = 3,38, p = 0,50). O problema é sistêmico, não regional.
- **Modelo preditivo (OLS):** com atraso, contatos com o SAC, tempo de resolução e valor do pedido, o modelo explica 53% da variação da nota no treino e 54% no teste. O erro médio absoluto é de 1,36 ponto, contra 2,07 de uma linha de base que prevê sempre a média. Nenhuma previsão chega à faixa de promotor (nota ≥ 9).

---

## 🛠️ Estrutura do Projeto

```text
├── .github/workflows/ci.yml   # Pipeline de CI: lint (flake8) e testes (pytest)
├── data/
│   └── desafio_nps_fase_1.csv # Base de dados de entrada
├── notebooks/
│   └── tech_challenge_andre_v2.ipynb  # Análise exploratória e modelagem
├── reports/
│   └── figures/               # Figuras geradas pelo pipeline
├── src/
│   ├── __init__.py
│   └── main.py                # Pipeline de análise (funções e execução)
├── tests/
│   └── test_main.py           # Teste de execução do pipeline
├── .dockerignore
├── .flake8                    # Limite de linha (88) para o flake8
├── .gitignore
├── .pre-commit-config.yaml    # isort, black e flake8 antes de cada commit
├── Dockerfile                 # Imagem isolada para executar o pipeline
├── environment.yml            # Ambiente Conda (fonte principal de dependências)
├── pytest.ini                 # Configuração do pytest
├── requirements.txt           # Dependências de execução (pip)
├── requirements-dev.txt       # Ferramentas de desenvolvimento (pip)
└── README.md
```

---

## ⚙️ Ambiente de Desenvolvimento (Conda)

O ambiente é definido em `environment.yml`, com todas as dependências, inclusive as ferramentas de qualidade, instaladas pelo conda. Não instale pacotes do mesmo ambiente com `pip`, para evitar conflitos entre os dois gerenciadores.

```bash
conda env create -f environment.yml
conda activate desafio_nps_fase_1
```

Para atualizar um ambiente já existente, recrie-o:

```bash
conda deactivate
conda env remove -n desafio_nps_fase_1
conda env create -f environment.yml
```

Para rodar o pipeline a partir da raiz do projeto:

```bash
python src/main.py
```

Os resultados são impressos no terminal e as figuras são salvas em `reports/figures/`.

---

## 🐳 Execução via Docker

O Docker reproduz o ambiente sem instalações locais.

**Pré-requisito:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e em execução.

1. **Clone o repositório:**

```bash
git clone https://github.com/gerferson-andre/desafio_nps_fase_1.git
cd desafio_nps_fase_1
```

2. **Construa a imagem:**

```bash
docker build -t tech-challenge-nps .
```

3. **Execute o pipeline e salve as figuras na pasta local `reports/`:**

```bash
docker run --rm -v "$(pwd)/reports:/app/reports" tech-challenge-nps
```

No PowerShell do Windows, use `${PWD}` no lugar de `$(pwd)`:

```powershell
docker run --rm -v "${PWD}/reports:/app/reports" tech-challenge-nps
```

4. **Execute a suíte de testes dentro do contêiner:**

```bash
docker run --rm tech-challenge-nps pytest
```

---

## 📈 Etapas do Pipeline

O arquivo `src/main.py` executa as etapas abaixo, nesta ordem.

1. **Diagnóstico da base:** dimensões, tipos, valores ausentes, duplicatas e estatística descritiva.
2. **Variável-alvo:** classificação em Detratores (nota < 7), Neutros (7 a 8,9) e Promotores (≥ 9), sem arredondamento, e cálculo do índice NPS.
3. **Normalidade:** teste de Shapiro-Wilk e gráficos (histograma com cortes e Q-Q plot).
4. **Associações:** coeficientes de Spearman entre cada variável e a nota, e matriz de correlação completa.
5. **Diagnóstico operacional:** comparação de médias entre a base total, os detratores e dois grupos de não detratores (com e sem recompra em 30 dias).
6. **Testes de grupo:**
   - Mann-Whitney U para recompra em 30 dias, com verificação de separação entre os grupos.
   - Kruskal-Wallis para região geográfica.
7. **Incerteza da média:** intervalo de confiança de 95% pela distribuição t de Student e por bootstrap (2.000 reamostras) sobre a base inteira.
8. **Modelo preditivo (OLS):** regressão linear com divisão de 80% para treino e 20% para teste, validação com métricas (MAE, RMSE e R²) e comparação com a linha de base.

---

## 🔒 Variáveis e Cuidados com Vazamento de Dados

- **Preditores do modelo:** atraso na entrega, contatos com o SAC, tempo de resolução e valor do pedido.
- **Excluídas do modelo:** `repeat_purchase_30d` (recompra em 30 dias) e `csat_internal_score` (satisfação interna). Ambas são medidas junto ou depois da pesquisa. Elas aparecem nas análises descritivas, sinalizadas como tal.
- **A confirmar:** o momento em que contatos com o SAC e tempo de resolução são registrados. Se forem posteriores à pesquisa, o modelo deve ser refeito sem essas variáveis.

---

## 🧪 Qualidade de Código e CI

- **Pre-commit** (`.pre-commit-config.yaml`): executa isort, black e flake8 a cada commit.
- **Limite de linha:** 88 caracteres, definido tanto em `.flake8` quanto no hook do pre-commit.
- **GitHub Actions** (`.github/workflows/ci.yml`): a cada push ou pull request, roda o flake8 sobre `src` e `tests` e executa a suíte com pytest.

Para rodar as verificações localmente:

```bash
flake8 src tests
pre-commit run --all-files
python -m pytest
```

---

## ⚠️ Limitações

- **Associação não é causa.** Os coeficientes e o modelo descrevem associações. Reduzir atrasos ou contatos com o SAC precisa ser testado antes de ser tratado como efeito garantido.
- **Base de um único período.** A base não tem datas, então os resultados descrevem estes 2.500 pedidos.
- **Separação entre recompra e nota.** Enquanto a origem dessa variável não for esclarecida, a comparação com recompra serve como diagnóstico.
- **Faixas de ação não definidas.** O projeto não estabelece os cortes de risco que a operação usaria.

---

## ✒️ Autor

**Gerferson André Silva Costa**

Pós-Graduação Tech em AI Scientist — FIAP
