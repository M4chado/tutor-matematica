"""Configurações centrais do projeto: caminhos, classes e limites.

Tudo que vários arquivos usam fica aqui, para mudar num lugar só.
"""
from pathlib import Path

# Pasta raiz do projeto (a pasta acima de src/)
RAIZ = Path(__file__).resolve().parent.parent

# Caminhos das pastas
PASTA_DADOS_BRUTOS = RAIZ / "data" / "raw"
PASTA_DADOS_PROCESSADOS = RAIZ / "data" / "processed"
PASTA_MODELOS = RAIZ / "models"
PASTA_FIGURAS = RAIZ / "reports" / "figuras"

# LLM local (Ollama). Usamos a versão "instruct", que responde direto,
# sem a etapa de "pensar" (a versão qwen3:4b comum sempre pensa, mesmo com think=False)
MODELO_LLM = "qwen3:4b-instruct"

# Classes, mapeamento das pastas e limites de confiança
# serão definidos nas próximas etapas (Etapa 1 e Etapa 6).
