"""Verifica se o computador está pronto para o projeto.

Como rodar (com o .venv ativado, na pasta do projeto):
    python src/verificar_ambiente.py

Cada item mostra ✅ (tudo certo) ou ❌ (precisa de atenção) com uma dica.
"""
import importlib
import os
import sys

# Esconde as mensagens técnicas do TensorFlow para a saída ficar limpa
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Garante que os símbolos ✅/❌ apareçam certo no terminal do Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Permite importar o config.py estando na pasta src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELO_LLM, PASTA_DADOS_BRUTOS  # noqa: E402

# Guarda o resultado de cada verificação: (nome, passou?, detalhe)
resultados = []


def registrar(nome, passou, detalhe=""):
    """Mostra o resultado na tela e guarda para o resumo final."""
    icone = "✅" if passou else "❌"
    print(f"{icone} {nome}" + (f" — {detalhe}" if detalhe else ""))
    resultados.append((nome, passou))


def verificar_python():
    """O projeto foi planejado para o Python 3.11."""
    versao = sys.version_info
    texto = f"{versao.major}.{versao.minor}.{versao.micro}"
    registrar("Python 3.11", (versao.major, versao.minor) == (3, 11), texto)


def verificar_biblioteca(nome_exibicao, nome_modulo, nome_pacote=None):
    """Tenta importar uma biblioteca e mostra a versão."""
    try:
        modulo = importlib.import_module(nome_modulo)
        versao = getattr(modulo, "__version__", None)
        if versao is None:
            # Algumas bibliotecas não guardam a versão no módulo; pergunta ao pip
            from importlib.metadata import version

            versao = version(nome_pacote or nome_modulo)
        registrar(nome_exibicao, True, f"versão {versao}")
    except Exception as erro:
        registrar(nome_exibicao, False, f"não importou ({erro}). Rode: pip install -r requirements.txt")


def verificar_webcam():
    """Tenta abrir a webcam e capturar uma imagem."""
    try:
        import cv2

        # CAP_DSHOW é o modo de câmera mais rápido e estável no Windows
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        conseguiu, quadro = camera.read()
        camera.release()
        if conseguiu and quadro is not None:
            altura, largura = quadro.shape[:2]
            registrar("Webcam", True, f"imagem de {largura}x{altura} capturada")
        else:
            registrar("Webcam", False, "não abriu. Feche outros apps que usam a câmera (Teams, Zoom...)")
    except Exception as erro:
        registrar("Webcam", False, str(erro))


def verificar_ollama():
    """Confere se o Ollama está rodando, se o modelo existe e se ele responde."""
    try:
        import ollama

        resposta = ollama.list()
    except Exception as erro:
        registrar("Ollama respondendo", False, f"sem resposta ({erro}). Abra o aplicativo Ollama")
        registrar(f"Modelo {MODELO_LLM} instalado", False, "não deu para verificar sem o Ollama")
        return

    registrar("Ollama respondendo", True)

    # A lista de modelos pode vir como objeto ou dicionário, conforme a versão
    modelos = getattr(resposta, "models", None) or resposta.get("models", [])
    nomes = [getattr(m, "model", None) or m.get("model") or m.get("name") for m in modelos]
    if MODELO_LLM not in nomes:
        registrar(f"Modelo {MODELO_LLM} instalado", False, f"não encontrado. Rode: ollama pull {MODELO_LLM}")
        return

    # Faz uma pergunta bem curta para provar que o modelo carrega e responde
    try:
        import time

        inicio = time.time()
        conversa = ollama.chat(
            model=MODELO_LLM,
            messages=[{"role": "user", "content": "Responda apenas: ok"}],
            think=False,  # pensamento desligado: respostas mais rápidas
            options={"num_predict": 10},
        )
        segundos = time.time() - inicio
        texto = conversa.message.content.strip()
        # Se a resposta for longa, o modelo "pensou em voz alta" mesmo com think=False
        if len(texto) <= 15:
            registrar(f"Modelo {MODELO_LLM} instalado", True, f'respondeu "{texto}" em {segundos:.1f}s')
        else:
            registrar(
                f"Modelo {MODELO_LLM} instalado",
                False,
                f'responde, mas o pensamento vaza na resposta ("{texto[:40]}..."). '
                "Use uma versão sem pensamento (ex.: qwen3:4b-instruct)",
            )
    except Exception as erro:
        registrar(f"Modelo {MODELO_LLM} instalado", False, f"instalado, mas não respondeu ({erro})")


def verificar_dataset():
    """Confere se o dataset do Kaggle foi extraído em data/raw/."""
    if not PASTA_DADOS_BRUTOS.exists():
        registrar("Dataset em data/raw/", False, "pasta não existe")
        return

    # Procura as pastas de classes (podem estar dentro de uma subpasta)
    pastas = [p for p in PASTA_DADOS_BRUTOS.rglob("*") if p.is_dir()]
    pastas_com_imagens = [p for p in pastas if any(p.glob("*.jpg")) or any(p.glob("*.png"))]
    if pastas_com_imagens:
        registrar("Dataset em data/raw/", True, f"{len(pastas_com_imagens)} pastas de símbolos encontradas")
    else:
        registrar("Dataset em data/raw/", False, "nenhuma imagem encontrada. Extraia o zip do Kaggle aqui")


def main():
    print("\n=== Verificando o ambiente do Tutor de Matemática ===\n")
    verificar_python()
    verificar_biblioteca("TensorFlow", "tensorflow")
    verificar_biblioteca("OpenCV", "cv2")
    verificar_biblioteca("SymPy", "sympy")
    verificar_biblioteca("Gradio", "gradio")
    verificar_biblioteca("ollama (biblioteca Python)", "ollama")
    verificar_webcam()
    verificar_ollama()
    verificar_dataset()

    aprovados = sum(1 for _, passou in resultados if passou)
    total = len(resultados)
    print(f"\nResultado: {aprovados}/{total} itens OK.")
    if aprovados == total:
        print("Tudo pronto! 🎉")
    else:
        print("Corrija os itens com ❌ e rode de novo.")


if __name__ == "__main__":
    main()
