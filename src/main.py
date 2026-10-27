"""Análise de NPS: Tech Challenge Fase 1.

Espelha o notebook tech_challenge_andre_v4.ipynb (seções 1 a 9.2 e 10).
Execute a partir da raiz do projeto:  python src/main.py

Saídas: prints no terminal e figuras em reports/figures/.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")  # execução sem interface gráfica (Docker/CI)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "desafio_nps_fase_1.csv")
FIG_DIR = os.path.join("reports", "figures")

IDS = ["customer_id", "order_id"]

# Medidas junto ou depois da pesquisa: aparecem na análise, mas não no modelo
POS_PESQUISA = ["csat_internal_score", "repeat_purchase_30d"]

FEATURES = [
    "delivery_delay_days",
    "customer_service_contacts",
    "resolution_time_days",
    "order_value",
]

INDICADORES = {
    "delivery_delay_days": "Atraso na entrega (dias)",
    "customer_service_contacts": "Contatos no SAC (qtd)",
    "resolution_time_days": "Tempo de resolução (dias)",
    "complaints_count": "Reclamações (qtd)",
    "order_value": "Valor do pedido (R$)",
    "freight_value": "Valor do frete (R$)",
}

ORDEM_CATEGORIAS = ["Detrator", "Neutro", "Promotor"]
SEG_DETRATOR = "Detratores"
SEG_RECOMPRA = "Não detratores com recompra"
SEG_SEM_RECOMPRA = "Não detratores sem recompra"
ORDEM_SEGMENTOS = [SEG_DETRATOR, SEG_RECOMPRA, SEG_SEM_RECOMPRA]


# ---------------------------------------------------------------------------
# 1 e 2. Dados e diagnóstico
# ---------------------------------------------------------------------------
def carregar_dados():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def diagnosticar_base(df):
    print("=== 2. DIAGNÓSTICO DA BASE ===")
    df.info()
    print(f"\nLinhas: {df.shape[0]} | Colunas: {df.shape[1]}")
    print(
        f"Valores ausentes: {int(df.isna().sum().sum())} | "
        f"Linhas duplicadas: {int(df.duplicated().sum())}"
    )
    print(
        f"order_id únicos: {df['order_id'].nunique()} | "
        f"customer_id únicos: {df['customer_id'].nunique()}\n"
    )
    resumo = df.drop(columns=IDS).describe().T.round(2)
    print(resumo.to_string())


# ---------------------------------------------------------------------------
# 3. Variável-alvo e régua do NPS
# ---------------------------------------------------------------------------
def categoria_nps(nota):
    """Classifica a nota pela régua oficial, sem arredondamento.

    Detrator < 7, Neutro de 7 a 8,9 e Promotor >= 9.
    """
    return np.select([nota < 7, nota >= 9], ["Detrator", "Promotor"], default="Neutro")


def classificar_nps(df):
    print("\n=== 3. VARIÁVEL-ALVO E RÉGUA DO NPS ===")
    nps = df["nps_score"]
    if nps.isna().any():
        raise ValueError("nps_score tem valores ausentes")
    if not nps.between(0, 10).all():
        raise ValueError("nps_score está fora da escala de 0 a 10")

    df["nps_categoria"] = pd.Categorical(
        categoria_nps(nps), categories=ORDEM_CATEGORIAS, ordered=True
    )
    contagem = df["nps_categoria"].value_counts().reindex(ORDEM_CATEGORIAS)
    percentual = contagem / len(df) * 100
    print(pd.DataFrame({"n": contagem, "%": percentual}).round(2).to_string())

    indice = percentual["Promotor"] - percentual["Detrator"]
    print(f"\nÍndice NPS = %Promotores − %Detratores = {indice:.2f}")
    print(f"Nota média: {nps.mean():.2f} | Mediana: {nps.median():.2f}")


# ---------------------------------------------------------------------------
# 4. Normalidade
# ---------------------------------------------------------------------------
def testar_normalidade(nps):
    print("\n=== 4. NORMALIDADE DA NOTA ===")
    stat_w, p_w = stats.shapiro(nps)
    print(f"Shapiro-Wilk: W = {stat_w:.4f} | p-valor = {p_w:.4g}")
    if p_w < 0.05:
        print("Normalidade rejeitada (p < 0,05).")
    else:
        print("Não há evidência contra a normalidade.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(nps, bins=np.arange(0, 10.5, 0.5), kde=True, color="teal", ax=axes[0])
    for corte in (7, 9):
        axes[0].axvline(corte, ls="--", color="grey")
    axes[0].set_title("Distribuição da nota (linhas em 7 e 9)", fontweight="bold")
    axes[0].set_xlabel("Nota de NPS (0 a 10)")

    stats.probplot(nps, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q plot contra a normal teórica", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "grafico_distribuicao.png"), bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Associações (Spearman)
# ---------------------------------------------------------------------------
def cor_da_barra(variavel, rho):
    if variavel in POS_PESQUISA:
        return "#c9cedb"
    return "#e74c3c" if rho < 0 else "#2ecc71"


def analisar_correlacoes(df):
    print("\n=== 5. ASSOCIAÇÃO COM A NOTA (SPEARMAN) ===")
    cols_num = [c for c in df.select_dtypes(include=np.number).columns if c not in IDS]
    matriz = df[cols_num].corr(method="spearman")

    corr_nps = matriz["nps_score"].drop("nps_score").sort_values()
    print(corr_nps.round(4).to_string())

    cores = [cor_da_barra(v, r) for v, r in corr_nps.items()]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(corr_nps.index, corr_nps.values, color=cores)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Coeficiente de Spearman (ρ) com nps_score")
    ax.set_title(
        "Associação de cada variável com a nota de NPS",
        fontweight="bold",
        loc="left",
    )
    for i, v in enumerate(corr_nps.values):
        ax.text(
            v + (0.02 if v >= 0 else -0.02),
            i,
            f"{v:+.2f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "grafico_correlacao.png"), bbox_inches="tight")
    plt.close()

    mascara = np.triu(np.ones_like(matriz, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matriz,
        mask=mascara,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "ρ de Spearman"},
    )
    plt.title(
        "Matriz de Spearman entre variáveis numéricas",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "matriz_correlacao.png"), bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 6. Diagnóstico operacional por segmento
# ---------------------------------------------------------------------------
def criar_segmentos(df):
    segmentos = np.select(
        [df["nps_score"] < 7, df["repeat_purchase_30d"] == 1],
        [SEG_DETRATOR, SEG_RECOMPRA],
        default=SEG_SEM_RECOMPRA,
    )
    df["segmento_diagnostico"] = pd.Categorical(
        segmentos, categories=ORDEM_SEGMENTOS, ordered=True
    )


def tabela_segmentos(df):
    print("\n=== 6. DIAGNÓSTICO OPERACIONAL: DETRATORES E NÃO DETRATORES ===")
    colunas = list(INDICADORES)
    medias = df.groupby("segmento_diagnostico", observed=True)[colunas].mean().T
    medias.insert(0, "Base total", df[colunas].mean())
    tabela = medias[["Base total"] + ORDEM_SEGMENTOS].rename(index=INDICADORES)

    n_por_grupo = df["segmento_diagnostico"].value_counts().reindex(ORDEM_SEGMENTOS)
    tabela.loc["n (clientes)"] = [len(df)] + n_por_grupo.tolist()
    print(tabela.round(2).to_string())


# ---------------------------------------------------------------------------
# 7. Testes de grupo
# ---------------------------------------------------------------------------
def testar_grupos(df):
    print("\n=== 7. TESTES DE GRUPO ===")

    g_sem = df.loc[df["repeat_purchase_30d"] == 0, "nps_score"]
    g_com = df.loc[df["repeat_purchase_30d"] == 1, "nps_score"]
    print("-- 7.1 Recompra em 30 dias (Mann-Whitney U) --")
    print(
        f"Nota média sem recompra: {g_sem.mean():.2f} | "
        f"com recompra: {g_com.mean():.2f}"
    )
    print(
        f"Maior nota sem recompra: {g_sem.max():.1f} | "
        f"menor nota com recompra: {g_com.min():.1f}"
    )
    print(f"Distância entre os grupos: {g_com.min() - g_sem.max():.1f} ponto(s)")
    stat_mw, p_mw = stats.mannwhitneyu(g_sem, g_com, alternative="two-sided")
    print(f"Mann-Whitney U = {stat_mw:.1f} | p-valor = {p_mw:.4g}")
    print(
        "ATENÇÃO: separação total entre os grupos; investigar a origem da variável.\n"
    )

    grupos_regiao = [
        g["nps_score"].to_numpy() for _, g in df.groupby("customer_region")
    ]
    stat_kw, p_kw = stats.kruskal(*grupos_regiao)
    print("-- 7.2 Região do cliente (Kruskal-Wallis) --")
    print(f"Kruskal-Wallis H = {stat_kw:.4f} | p-valor = {p_kw:.4f}")
    if p_kw > 0.05:
        print("Não há evidência de diferença de nota entre regiões (p > 0,05).")
    else:
        print("Há evidência de diferença de nota entre regiões (p < 0,05).")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(
        x="repeat_purchase_30d",
        y="nps_score",
        data=df,
        hue="repeat_purchase_30d",
        palette=["#e74c3c", "#2ecc71"],
        legend=False,
        ax=axes[0],
    )
    axes[0].set_xticks([0, 1], ["Não recomprou", "Recomprou"])
    axes[0].set_title(
        "Nota por recompra em 30 dias\n(medida após a pesquisa)",
        fontweight="bold",
    )
    axes[0].set_xlabel("Recompra em 30 dias")
    axes[0].set_ylabel("Nota de NPS")

    sns.boxplot(
        x="customer_region",
        y="nps_score",
        data=df,
        hue="customer_region",
        palette="Set2",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Nota por região", fontweight="bold")
    axes[1].set_xlabel("Região do cliente")
    axes[1].set_ylabel("Nota de NPS")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "boxplots_comparativos.png"), bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 8. Intervalo de confiança da média
# ---------------------------------------------------------------------------
def intervalo_confianca(df):
    print("\n=== 8. INCERTEZA DA MÉDIA DA NOTA ===")
    rng = np.random.default_rng(RANDOM_STATE)
    nota = df["nps_score"].to_numpy()
    media = nota.mean()

    ic_t = stats.t.interval(0.95, df=len(nota) - 1, loc=media, scale=stats.sem(nota))

    n_boot = 2000
    reamostras = rng.choice(nota, size=(n_boot, len(nota)), replace=True)
    ic_boot = np.percentile(reamostras.mean(axis=1), [2.5, 97.5])

    print(f"Média da nota: {media:.4f}")
    print(f"IC 95% (t de Student): ({ic_t[0]:.4f}, {ic_t[1]:.4f})")
    print(
        f"IC 95% (bootstrap, {n_boot} reamostras): "
        f"({ic_boot[0]:.4f}, {ic_boot[1]:.4f})"
    )


# ---------------------------------------------------------------------------
# 9. Modelo preditivo (OLS)
# ---------------------------------------------------------------------------
def metricas(real, previsto):
    return {
        "MAE": mean_absolute_error(real, previsto),
        "RMSE": np.sqrt(mean_squared_error(real, previsto)),
        "R²": r2_score(real, previsto),
    }


def ajustar_modelo(df):
    print("\n=== 9.1 AJUSTE DA REGRESSÃO LINEAR (OLS) ===")
    X = df[FEATURES]
    y = df["nps_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    modelo = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    print(modelo.summary())

    print("\n=== 9.2 VALIDAÇÃO NO CONJUNTO DE TESTE ===")
    y_pred = modelo.predict(sm.add_constant(X_test))
    fora = ((y_pred < 0) | (y_pred > 10)).mean()
    print(f"Previsões fora de [0, 10] antes do ajuste: {fora:.1%}")

    y_pred_ajustado = y_pred.clip(0, 10)
    promotor = (y_pred_ajustado >= 9).mean()
    print(f"Previsões na faixa de promotor (≥ 9): {promotor:.1%}")

    baseline = np.full(len(y_test), y_train.mean())
    comparacao = pd.DataFrame(
        {
            "Modelo OLS": metricas(y_test, y_pred_ajustado),
            "Linha de base (média)": metricas(y_test, baseline),
        }
    )
    print(comparacao.round(4).to_string())


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    pd.set_option("display.float_format", "{:.2f}".format)

    df = carregar_dados()
    diagnosticar_base(df)
    classificar_nps(df)
    testar_normalidade(df["nps_score"])
    analisar_correlacoes(df)
    criar_segmentos(df)
    tabela_segmentos(df)
    testar_grupos(df)
    intervalo_confianca(df)
    ajustar_modelo(df)
    print("\nAnálise concluída.")


if __name__ == "__main__":
    main()
