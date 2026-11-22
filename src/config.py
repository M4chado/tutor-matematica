"""Configurações centrais do projeto: caminhos, classes e limites.

Tudo que vários arquivos usam fica aqui, para mudar num lugar só.
"""
from pathlib import Path

# Pasta raiz do projeto (a pasta acima de src/)
RAIZ = Path(__file__).resolve().parent.parent

# Caminhos das pastas
PASTA_DADOS_BRUTOS = RAIZ / "data" / "raw"
PASTA_IMAGENS_BRUTAS = PASTA_DADOS_BRUTOS / "extracted_images"  # uma subpasta por símbolo
PASTA_DADOS_PROCESSADOS = RAIZ / "data" / "processed"
PASTA_MODELOS = RAIZ / "models"
PASTA_FIGURAS = RAIZ / "reports" / "figuras"

# ---------------------------------------------------------------------------
# Classes (Etapa 1)
# ---------------------------------------------------------------------------
# Mapeamento aprovado pelo usuário: símbolo -> nome da pasta no dataset.
# A ordem aqui define o número de cada classe (0 a 16) dentro da rede.
MAPA_PASTAS = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "+": "+",
    "-": "-",
    "×": "times",
    "÷": "div",
    "=": "=",
    "(": "(",
    ")": ")",
    "x": "X",   # a pasta "X" contém o x minúsculo escrito à mão (incógnita)
}
CLASSES = list(MAPA_PASTAS.keys())  # 17 classes

# ---------------------------------------------------------------------------
# Preparação dos dados (Etapa 1)
# ---------------------------------------------------------------------------
TAMANHO_IMAGEM = 45          # as imagens viram quadrados de 45x45 pixels
LIMITE_POR_CLASSE = 3000     # no máximo 3.000 imagens por classe (equilíbrio)
PROPORCAO_TREINO = 0.70
PROPORCAO_VALIDACAO = 0.15
PROPORCAO_TESTE = 0.15
SEMENTE = 42                 # semente fixa: o sorteio sai sempre igual

# ---------------------------------------------------------------------------
# LLM local (Ollama)
# ---------------------------------------------------------------------------
# Usamos a versão "instruct", que responde direto, sem a etapa de "pensar"
# (a versão qwen3:4b comum sempre pensa, mesmo com think=False)
MODELO_LLM = "qwen3:4b-instruct"

# ---------------------------------------------------------------------------
# Confiança da CNN
# ---------------------------------------------------------------------------
# Abaixo deste valor, o palpite é considerado "duvidoso": a caixa fica laranja
# na tela e o agente pergunta ao visitante antes de calcular.
# Calibrado na Etapa 6 com o conjunto de VALIDAÇÃO (src/calibrar_confianca.py):
# com 0,80 o agente pergunta em 4,1% dos símbolos e os erros que passam sem
# pergunta caem de 2,5% para 1,1%. Acima de 0,90 as perguntas disparam.
LIMITE_CONFIANCA = 0.80
