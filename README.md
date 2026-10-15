Aqui está o **README.md** reformulado com foco total na execução e reprodução do projeto **via Docker** e **Docker Compose**, simplificando o fluxo para qualquer pessoa rodar a análise sem precisar configurar ambientes Python locais:

---

```markdown
# 📊 Tech Challenge — Análise Preditiva e Diagnóstico de NPS

Este repositório contém a solução do **Tech Challenge (Fase 1)** da Pós-Tech em AI Scientist (FIAP). O projeto consiste em uma análise estatística e econométrica avançada sobre a base de dados de **Net Promoter Score (NPS)** de uma operação de e-commerce, visando transitar de uma gestão reativa de satisfação para um modelo preditivo focado em alavancas operacionais e logísticas.

---

## 📌 Sumário Executivo

A análise diagnosticou uma base com alto índice de insatisfação (**NPS Index Score de -66.00**), fortemente correlacionada com gargalos de atendimento (SAC) e atrasos logísticos. Através de métodos paramétricos e não paramétricos, além de regressão OLS, identificaram-se os principais *drivers* de detratores e oportunidades de retenção (recompra).

### Principais Achados
- **Distribuição do NPS**: Média de $4.38 \pm 2.51$, com $74.04\%$ de detratores, $17.92\%$ de neutros e $8.04\%$ de promotores.
- **Normalidade Rejeitada**: O teste de Shapiro-Wilk ($W = 0.9817, p < 0.0001$) confirmou que o NPS não segue distribuição normal, direcionando a análise para inferência não paramétrica.
- **Drivers de Insatisfação**: As variáveis com maior correlação negativa ($r_s$ de Spearman) com o NPS foram a **quantidade de reclamações** e o **tempo de resolução no SAC**.
- **Impacto na Recompra**: O teste de Mann-Whitney U confirmou com significância estatística ($p < 0.0001$) que clientes da *Coorte de Sucesso* (que realizam recompra em até 30 dias) possuem notas de NPS significativamente mais altas.
- **Efeito Regional**: O teste de Kruskal-Wallis ($p > 0.05$) indicou ausência de diferença estatisticamente significante de NPS entre as regiões geográficas dos clientes.

---

## 🛠️ Estrutura do Projeto

```text
├── .github/workflows/   # Esteira de Integração Contínua (CI/CD)
├── data/                # Base de dados de entrada (CSV)
│   └── desafio_nps_fase_1.csv
├── reports/             # Relatórios e exportação de gráficos (mapeado via volume no Docker)
│   └── figures/
├── src/                 # Código-fonte da aplicação
│   ├── __init__.py
│   └── main.py          # Script principal de execução da pipeline de análise
├── tests/               # Suíte de testes unitários (pytest)
├── Dockerfile           # Imagem isolada do ambiente de execução Python
├── docker-compose.yml   # Orquestração do contêiner e mapeamento de volumes
├── .dockerignore        # Arquivos ignorados na build do contêiner
├── environment.yml      # Especificação das dependências Conda/Python
└── README.md            # Documentação do projeto

```

---

## 🐳 Execução via Docker (Recomendado)

Todo o ambiente de dependências (Python, Pandas, SciPy, Scikit-Learn, Statsmodels, Matplotlib, Seaborn) foi conteinerizado para garantir reprodutibilidade total sem a necessidade de instalações locais.

### Pré-requisitos

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado e em execução.

---

### Opção 1: Execução Simplificada com Docker Compose

Esta é a opção mais prática, pois já configura automaticamente os volumes para que os gráficos gerados no contêiner sejam salvos diretamente na pasta local `reports/figures/`.

1. **Clone o repositório:**
```bash
git clone [https://github.com/seu-usuario/tech-challenge-fase1-nps.git](https://github.com/seu-usuario/tech-challenge-fase1-nps.git)
cd tech-challenge-fase1-nps

```


2. **Execute a pipeline completa:**
```bash
docker-compose up --build

```



---

### Opção 2: Execução Direta via Docker CLI

Caso prefira utilizar diretamente os comandos do Docker:

1. **Construir a imagem:**
```bash
docker build -t tech-challenge-nps .

```


2. **Rodar o contêiner com mapeamento da pasta de relatórios:**
```bash
docker run --rm -v $(pwd)/reports:/app/reports tech-challenge-nps

```


*(No PowerShell do Windows, substitua `$(pwd)` por `${PWD}`)*
3. **Executar a suíte de testes dentro do contêiner:**
```bash
docker run --rm tech-challenge-nps pytest

```



---

## 📈 Etapas da Pipeline Executada no Contêiner

1. **Sanitização e Diagnóstico de Dados**: leitura, checagem de tipos, detecção de nulos/duplicados e estatística descritiva completa das variáveis numéricas.
2. **Análise Descritiva do NPS**: cálculo de centro, dispersão, assimetria/curtose e classificação oficial (Detratores, Neutros, Promotores).
3. **Diagnóstico SLA e Segmentação**: tabela comparativa de médias operacionais entre a média geral da base, detratores e a *Coorte de Sucesso*.
4. **Testes Formais de Normalidade**: avaliação por Shapiro-Wilk e exportação de gráficos (Histograma/KDE e Q-Q Plot).
5. **Análise de Correlação e Heatmap**: coeficientes de correlação de Spearman para identificação de força e direção das variáveis em relação ao NPS.
6. **Inferência Estatística e Amostragem**: construção de Intervalos de Confiança a 95% via t-Student e Bootstrap (2.000 iterações).
7. **Testes de Hipótese Não Paramétricos**:
* **Mann-Whitney U**: avaliação de diferenças de NPS em função da recompra em 30 dias (`repeat_purchase_30d`).
* **Kruskal-Wallis**: avaliação de variabilidade de NPS entre regiões geográficas (`customer_region`).


8. **Modelagem Preditiva Econométrica (OLS)**: ajuste de modelo de regressão linear para explicação e predição da nota de NPS com métricas de avaliação ($R^2$, MAE, RMSE) no conjunto de teste.

---

## 🧪 Qualidade de Código e CI/CD

O repositório está integrado com:

* **Testes Unitários**: executados via `pytest` validando os módulos e transformações.
* **Hooks de Pre-commit**: garantia de formatação e padrões de código.
* **GitHub Actions**: esteira de CI em `.github/workflows/ci.yml` que valida o build da imagem Docker e executa a suíte de testes automatizados a cada *push* ou *pull request*.

---

## ✒️ Autor

**Gerferson André Silva Costa**

Pós-Graduação Tech em AI Scientist — FIAP

```

```