# Imagem base oficial e enxuta do Python
FROM python:3.10-slim

# Diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências para o container
COPY requirements.txt .

# Instala as bibliotecas listadas no requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código e dados do projeto para o container
COPY . .

# Comando padrão para executar o projeto
CMD ["python", "src/main.py"]