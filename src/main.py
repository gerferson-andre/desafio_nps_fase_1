import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Configuração não interativa do matplotlib para Docker/CI
matplotlib.use("Agg")


def processar_analise():
    print("--- A iniciar a leitura e processamento dos dados ---")

    # 1. Definição da base de dados (Acessando a pasta data/)
    caminho_dados = os.path.join("data", "desafio_nps_fase_1.csv")

    try:
        df = pd.read_csv(caminho_dados)
        print("Dados carregados com sucesso!\n")
    except FileNotFoundError:
        print(f"Erro: Ficheiro '{caminho_dados}' não encontrado.")
        return

    # Garantir criação do diretório de figuras
    os.makedirs(os.path.join("reports", "figures"), exist_ok=True)

    # 2. Diagnóstico de Dimensões e Tipagem
    print("=== ETAPA: DIMENSÕES E TIPAGEM DOS DADOS ===")
    print(
        f"Linhas (Observações): {df.shape[0]} | "
        f"Colunas (Atributos): {df.shape[1]}\n"
    )
    print(df.info())

    # 3. Verificação de Dados Ausentes (Nulos/BRANCOS) e Duplicatas
    print("\n=== ETAPA: SANITIZAÇÃO E QUALIDADE DOS DADOS ===")
    nulos = df.isnull().sum()
    msg_nulos = f"Valores Ausentes na Target (nps_score): {nulos['nps_score']}"
    print(msg_nulos)
    print(f"Total de Registros Duplicados: {df.duplicated().sum()}")

    # 4. Sumário estatístico das variáveis (excluindo IDs)
    print("\n=== ETAPA: SUMÁRIO ESTATÍSTICO DAS VARIÁVEIS ===")
    colunas_ignoradas = ["customer_id", "order_id"]
    tabela_descritiva = df.drop(columns=colunas_ignoradas).describe().T.round(2)
    print(tabela_descritiva)

    sns.set_theme(style="whitegrid")

    # 5. Estatística Descritiva da Target
    nps = df["nps_score"]
    print("\n=== MEDIDAS DESCRITIVAS DO NPS SCORE ===")
    print(
        f"Média: {nps.mean():.4f} | Mediana: {nps.median():.4f} | "
        f"Desvio Padrão: {nps.std():.4f}"
    )
    print(
        f"Q1 (25%): {nps.quantile(0.25):.4f} | "
        f"Q3 (75%): {nps.quantile(0.75):.4f} | "
        f"IQR: {nps.quantile(0.75) - nps.quantile(0.25):.4f}"
    )
    print(
        f"Assimetria (Skewness): {nps.skew():.4f} | " f"Curtose: {nps.kurtosis():.4f}\n"
    )

    # 6. Categorização Oficial do NPS
    detratores = (nps <= 6.0).sum()
    neutros = ((nps > 6.0) & (nps <= 8.0)).sum()
    promotores = (nps > 8.0).sum()
    total = len(nps)
    pct_det = (detratores / total) * 100
    pct_neu = (neutros / total) * 100
    pct_pro = (promotores / total) * 100
    nps_index = pct_pro - pct_det

    print("--- CLASSIFICAÇÃO DA BASE DE CLIENTES ---")
    print(f"Detratores (notas <= 6.0): {detratores} ({pct_det:.2f}%)")
    print(f"Neutros (notas 6.0 - 8.0]: {neutros} ({pct_neu:.2f}%)")
    print(f"Promotores (notas > 8.0): {promotores} ({pct_pro:.2f}%)")
    print(f"NPS INDEX SCORE (%Promotores - %Detratores): {nps_index:.2f}\n")

    # 7. Definição dos segmentos para fins descritivos
    df["segmento_diagnostico"] = "Demais Clientes"
    df.loc[df["nps_score"] <= 6.0, "segmento_diagnostico"] = "Detratores (NPS 0-6)"
    df.loc[
        (df["nps_score"] >= 9.0) & (df["repeat_purchase_30d"] == 1),
        "segmento_diagnostico",
    ] = "Coorte de Sucesso (NPS>=9 + Recompra)"

    # 8. Seleção dos indicadores operacionais e logísticos chave
    indicadores_operacionais = [
        "delivery_delay_days",
        "customer_service_contacts",
        "resolution_time_days",
        "complaints_count",
        "order_value",
        "freight_value",
    ]

    # 9. Construção da tabela comparativa de médias
    tabela_sla = df.groupby("segmento_diagnostico")[indicadores_operacionais].mean().T
    tabela_sla["Média Geral da Base"] = df[indicadores_operacionais].mean()

    # 10. Reordenação das colunas para facilidade de leitura executiva
    tabela_sla = tabela_sla[
        [
            "Média Geral da Base",
            "Detratores (NPS 0-6)",
            "Coorte de Sucesso (NPS>=9 + Recompra)",
        ]
    ]

    tabela_sla.index = [
        "Atraso na Entrega (dias)",
        "Contatos no SAC (qtd)",
        "Tempo de Resolução SAC (dias)",
        "Qtd de Reclamações",
        "Valor do Pedido (R$)",
        "Valor do Frete (R$)",
    ]

    print("\n=== TABELA COMPARATIVA DE MÉDIAS ===")
    print(tabela_sla.round(2))

    # 11. Teste de Shapiro-Wilk
    stat_w, p_val_w = stats.shapiro(nps)
    print("\n=== TESTE FORMAL DE NORMALIDADE (SHAPIRO-WILK) ===")
    print(f"Estatística W = {stat_w:.4f} | p-valor = {p_val_w:.4f}")
    if p_val_w < 0.05:
        print(
            "❌ Rejeita-se H₀ (p < 0.05): "
            "A variável 'nps_score' NÃO segue distribuição normal."
        )
        print("-> Direcionamento Metodológico: " "Adotar MÉTODOS NÃO PARAMÉTRICOS.")

    # 12. Preparar os dados para gráficos
    nps_clean = pd.to_numeric(df["nps_score"], errors="coerce").dropna()
    nps_array = nps_clean.to_numpy(dtype=float)

    if len(nps_array) == 0:
        raise ValueError("A coluna nps_score não contém valores numéricos válidos.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(x=nps_array, bins=11, ax=axes[0], color="teal", alpha=0.6)

    if len(nps_array) > 1 and np.unique(nps_array).size > 1:
        kde = gaussian_kde(nps_array)
        x_grid = np.linspace(nps_array.min() - 0.5, nps_array.max() + 0.5, 300)
        y_kde = np.asarray(kde(x_grid), dtype=float)
        bin_width = (nps_array.max() - nps_array.min() + 1) / 11
        y_kde = y_kde * len(nps_array) * bin_width

        axes[0].plot(x_grid, y_kde, color="darkslategray", linewidth=2)

    axes[0].set_title(
        "Distribuição de Frequência do NPS Score",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Nota de NPS (0 a 10)")
    axes[0].set_ylabel("Frequência")
    axes[0].set_xlim(0, 10)

    stats.probplot(nps_array, dist="norm", plot=axes[1])
    axes[1].set_title(
        "Q-Q Plot: Comparação com a Normal Teórica",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].get_lines()[0].set_color("indigo")
    axes[1].get_lines()[1].set_color("red")

    plt.tight_layout()
    caminho_fig1 = os.path.join("reports", "figures", "grafico_distribuicao.png")
    plt.savefig(caminho_fig1, bbox_inches="tight")
    plt.close()

    # 13. Correlação de Spearman
    cols_num = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in ["customer_id", "order_id"]
    ]

    corr_nps = (
        df[cols_num]
        .corr(method="spearman")["nps_score"]
        .drop("nps_score")
        .sort_values()
    )

    print("\n=== RANKING DE CORRELAÇÃO DE SPEARMAN COM O NPS ===")
    print(corr_nps.round(4))

    plt.figure(figsize=(10, 7))
    cores = ["#e74c3c" if x < 0 else "#2ecc71" for x in corr_nps]
    sns.barplot(
        x=corr_nps.values,
        y=corr_nps.index,
        hue=corr_nps.index,
        palette=cores,
        legend=False,
    )

    plt.title(
        "Força de Associação (Correlação de Spearman) com o NPS",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Coeficiente de Correlação (r_s)")
    plt.ylabel("")
    plt.axvline(x=0, color="black", linewidth=1.2, linestyle="--")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    caminho_fig2 = os.path.join("reports", "figures", "grafico_correlacao.png")
    plt.savefig(caminho_fig2, bbox_inches="tight")
    plt.close()

    # 14. Matriz de Correlação Heatmap
    matriz_corr = df[cols_num].corr(method="spearman")
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matriz_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Coeficiente de Spearman (r_s)"},
    )
    plt.title(
        "Matriz de Correlação de Spearman — Variáveis Operacionais e NPS",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    caminho_fig3 = os.path.join("reports", "figures", "matriz_correlacao.png")
    plt.savefig(caminho_fig3, bbox_inches="tight")
    plt.close()

    # 15. Amostragem e Inferência Estatística
    np.random.seed(42)
    amostra_nps = df["nps_score"].sample(n=1000, replace=False)
    media_amostral = amostra_nps.mean()
    erro_padrao = stats.sem(amostra_nps)

    ic_tstudent = stats.t.interval(
        0.95, df=len(amostra_nps) - 1, loc=media_amostral, scale=erro_padrao
    )

    boot_means = [
        np.mean(np.random.choice(amostra_nps, size=len(amostra_nps), replace=True))
        for _ in range(2000)
    ]
    ic_bootstrap = np.percentile(boot_means, [2.5, 97.5])

    print("\n=== INFERÊNCIA ESTATÍSTICA E INTERVALOS DE CONFIANÇA (95%) ===")
    print(f"Média Amostral: {media_amostral:.4f}")
    print(f"IC t-Student (Paramétrico): {ic_tstudent}")
    print(f"IC Bootstrap (Não Paramétrico): {ic_bootstrap}")

    # 16. Testes de Hipótese
    g_sem_recompra = df[df["repeat_purchase_30d"] == 0]["nps_score"]
    g_com_recompra = df[df["repeat_purchase_30d"] == 1]["nps_score"]
    stat_mw, p_val_mw = stats.mannwhitneyu(g_sem_recompra, g_com_recompra)

    print("\n=== MANN-WHITNEY U (NPS vs RECOMPRA) ===")
    print(f"Média NPS sem Recompra (0): {g_sem_recompra.mean():.2f}")
    print(f"Média NPS com Recompra (1): {g_com_recompra.mean():.2f}")
    print(f"Estatística U = {stat_mw:.4f} | p-valor = {p_val_mw:.4f}")

    if p_val_mw < 0.05:
        print(
            "✅ Rejeita-se H₀: Clientes que realizam "
            "recompra possuem NPS significativamente maior.\n"
        )

    grupos_regioes = [
        grupo["nps_score"].values for _, grupo in df.groupby("customer_region")
    ]
    stat_kw, p_val_kw = stats.kruskal(*grupos_regioes)

    print("=== KRUSKAL-WALLIS (NPS vs REGIÃO) ===")
    print(f"Estatística H = {stat_kw:.4f} | p-valor = {p_val_kw:.4f}")

    if p_val_kw > 0.05:
        print(
            "❌ Não se rejeita H₀: Não há evidência "
            "estatística de diferença de NPS entre regiões.\n"
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(
        x="repeat_purchase_30d",
        y="nps_score",
        data=df,
        hue="repeat_purchase_30d",
        palette=["#e74c3c", "#2ecc71"],
        ax=axes[0],
    )
    axes[0].set_title(
        "NPS vs. Recompra em 30 dias\n(Mann-Whitney U: p < 0.001)",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Não recomprou", "recomprou"])
    axes[0].set_xlabel("Recompra em 30 dias")
    axes[0].set_ylabel("NPS Score")
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()

    sns.boxplot(
        x="customer_region",
        y="nps_score",
        data=df,
        hue="customer_region",
        palette="Set2",
        ax=axes[1],
    )
    axes[1].set_title("NPS por região geográfica", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Região do cliente")
    axes[1].set_ylabel("NPS Score")
    axes[1].tick_params(axis="x", rotation=45)
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()

    plt.tight_layout()
    caminho_fig4 = os.path.join("reports", "figures", "boxplots_comparativos.png")
    plt.savefig(caminho_fig4, bbox_inches="tight")
    plt.close()

    # 17. Regressão Linear OLS
    features = [
        "delivery_delay_days",
        "customer_service_contacts",
        "resolution_time_days",
        "order_value",
    ]
    X = df[features]
    y = df["nps_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)
    modelo_ols = sm.OLS(y_train, X_train_const).fit()

    print("\n=== SUMÁRIO ESTATÍSTICO DO MODELO OLS (TREINO) ===")
    print(modelo_ols.summary())

    y_pred = modelo_ols.predict(X_test_const)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n=== DESEMPENHO PREDITIVO NO CONJUNTO DE TESTE ===")
    print(f"MAE (Erro Médio Absoluto): {mae:.4f}")
    print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}")
    print(f"R² (Coeficiente de Determinação): {r2:.4f}")


if __name__ == "__main__":
    processar_analise()
