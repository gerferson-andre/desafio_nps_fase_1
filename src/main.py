import pandas as pd


def processar_analise():
    print("--- A iniciar a leitura e processamento dos dados ---")

    # 1. Leitura do ficheiro CSV presente na pasta do projeto
    try:
        df = pd.read_csv("data/desafio_nps_fase_1.csv")

        print("\nDados carregados com sucesso!")
        print(f"Total de registos encontrados: {len(df)}")
        print("\nPrimeiras 5 linhas do conjunto de dados:")
        print(df.head())

        # Aqui poderá adicionar futuramente a sua lógica de análise do NPS

    except FileNotFoundError:
        print("\nErro: Ficheiro 'desafio_nps_fase_1.csv' não encontrado.")


if __name__ == "__main__":
    processar_analise()
