from src.main import processar_analise


def test_processar_analise_execucao():
    """Testa se a função principal executa sem lançar exceções."""
    try:
        processar_analise()
        sucesso = True
    except Exception as e:
        sucesso = False
        print(f"Erro durante a execução: {e}")

    assert sucesso is True
