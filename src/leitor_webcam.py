"""Etapa 4: lê uma conta escrita no papel, pela webcam ou por uma foto.

Como rodar (com o .venv ativado, na pasta do projeto):
    python src/leitor_webcam.py                    # abre a webcam
    python src/leitor_webcam.py --imagem conta.jpg # lê uma foto (sem webcam)
    python src/leitor_webcam.py --camera 1         # usa outra câmera

Na janela da webcam:
    ESPAÇO  captura e lê a conta escrita na folha (mão e fundo são ignorados;
            se nenhuma folha for encontrada, lê só o retângulo-guia)
    Q / ESC fecha

Cada leitura é salva em capturas/ (foto original + foto com as caixas),
para você poder conferir depois os acertos e erros dos testes com pessoas.
"""
import argparse
import datetime
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from config import LIMITE_CONFIANCA, PASTA_MODELOS, RAIZ  # noqa: E402
from segmentacao import desenhar_leitura, ler_conta, mascara_da_folha, montar_expressao  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PASTA_CAPTURAS = RAIZ / "capturas"
JANELA_CAMERA = "CalcuLetra - camera (ESPACO = ler, Q = sair)"
JANELA_LEITURA = "CalcuLetra - leitura"
# Retângulo-guia: 80% da largura e 40% da altura, no centro da imagem
# (usado quando nenhuma folha é encontrada na imagem)
GUIA_LARGURA, GUIA_ALTURA = 0.80, 0.40
COR_GUIA = (40, 160, 255)    # laranja (BGR)
COR_FOLHA = (255, 160, 40)   # azul (BGR): contorno da folha encontrada


def carregar_modelo():
    import keras

    print("Carregando a CNN...")
    return keras.models.load_model(PASTA_MODELOS / "cnn_simbolos.keras")


def retangulo_guia(quadro):
    altura, largura = quadro.shape[:2]
    w, h = int(largura * GUIA_LARGURA), int(altura * GUIA_ALTURA)
    x, y = (largura - w) // 2, (altura - h) // 2
    return x, y, w, h


def ler_quadro(quadro, modelo):
    """Lê a conta de um quadro da webcam ou de uma foto (BGR).

    1. Procura a folha. Se achar, lê a imagem inteira, mas só a tinta cercada
       de papel: mão, roupa e o que estiver atrás são ignorados.
    2. Se não achar folha, lê só o retângulo-guia do centro (como na Etapa 4).

    Devolve (simbolos, imagem_anotada, onde) com onde = "folha" ou "retângulo-guia".
    """
    tela = quadro.copy()
    folha = mascara_da_folha(quadro, quadro.shape[:2])
    if folha is not None:
        simbolos = ler_conta(quadro, modelo, so_na_folha=True)
        contornos, _ = cv2.findContours((folha > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(tela, contornos, -1, COR_FOLHA, 3)
        return simbolos, desenhar_leitura(tela, simbolos), "folha"
    x, y, w, h = retangulo_guia(quadro)
    simbolos = ler_conta(quadro[y:y + h, x:x + w], modelo)
    cv2.rectangle(tela, (x, y), (x + w, y + h), COR_GUIA, 3)
    return simbolos, desenhar_leitura(tela, simbolos, deslocamento=(x, y)), "retângulo-guia"


def imprimir_leitura(simbolos):
    """Mostra no terminal a expressão e os detalhes de cada símbolo."""
    if not simbolos:
        print("\nNenhum símbolo encontrado. Escreva com caneta escura dentro do retângulo.")
        return
    print(f"\nExpressão lida:  {montar_expressao(simbolos)}")
    for i, s in enumerate(simbolos, start=1):
        alternativas = ", ".join(f"{sim} {100 * p:.0f}%" for sim, p in s["top3"][1:])
        aviso = "   ⚠️ duvidoso" if s["confianca"] < LIMITE_CONFIANCA else ""
        print(f"   {i:2d}. {s['simbolo']}  {100 * s['confianca']:5.1f}%   (outras: {alternativas}){aviso}")


def salvar_captura(original, anotada):
    PASTA_CAPTURAS.mkdir(exist_ok=True)
    carimbo = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for sufixo, imagem in (("original", original), ("leitura", anotada)):
        destino = PASTA_CAPTURAS / f"{carimbo}_{sufixo}.png"
        cv2.imencode(".png", imagem)[1].tofile(str(destino))  # funciona com acentos no caminho
    print(f"   (salvo em {PASTA_CAPTURAS.relative_to(RAIZ)}/{carimbo}_*.png)")


def ler_foto(caminho, modelo):
    """Modo sem webcam: lê a conta de uma foto."""
    with Image.open(caminho) as imagem:
        rgb = np.array(ImageOps.exif_transpose(imagem).convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    simbolos, anotada, onde = ler_quadro(bgr, modelo)
    print(f"\nLido em: {onde}")
    imprimir_leitura(simbolos)
    salvar_captura(bgr, anotada)
    # Mostra o resultado numa janela que cabe na tela
    escala = min(1.0, 1200 / anotada.shape[1], 800 / anotada.shape[0])
    cv2.imshow(JANELA_LEITURA, cv2.resize(anotada, None, fx=escala, fy=escala))
    print("\nAperte qualquer tecla na janela da imagem para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def usar_webcam(indice_camera, modelo):
    camera = cv2.VideoCapture(indice_camera, cv2.CAP_DSHOW)  # CAP_DSHOW: mais estável no Windows
    if not camera.isOpened():
        print("Não consegui abrir a webcam. Feche outros apps que usam a câmera e tente de novo.")
        return
    # Pede uma resolução maior (se a câmera não suportar, ela usa a máxima que tiver)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("\nWebcam aberta. Mostre a conta dentro do retângulo e aperte ESPAÇO. Q para sair.")

    while True:
        ok, quadro = camera.read()
        if not ok:
            print("A webcam parou de enviar imagens.")
            break
        x, y, w, h = retangulo_guia(quadro)
        tela = quadro.copy()
        cv2.rectangle(tela, (x, y), (x + w, y + h), COR_GUIA, 2)
        cv2.putText(tela, "Mostre a folha com a conta aqui", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COR_GUIA, 2)
        cv2.imshow(JANELA_CAMERA, tela)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), ord("Q"), 27):  # 27 = ESC
            break
        if cv2.getWindowProperty(JANELA_CAMERA, cv2.WND_PROP_VISIBLE) < 1:  # fechou no X
            break
        if tecla == 32:  # ESPAÇO
            # Lê a tinta que está na folha (ou o retângulo-guia, se não achar a folha)
            simbolos, anotada, onde = ler_quadro(quadro, modelo)
            print(f"\nLido em: {onde}")
            imprimir_leitura(simbolos)
            salvar_captura(quadro, anotada)
            cv2.imshow(JANELA_LEITURA, anotada)

    camera.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Lê uma conta escrita à mão.")
    parser.add_argument("--imagem", type=Path, help="lê uma foto em vez de usar a webcam")
    parser.add_argument("--camera", type=int, default=0, help="número da câmera (padrão: 0)")
    args = parser.parse_args()

    modelo = carregar_modelo()
    if args.imagem:
        if not args.imagem.exists():
            print(f"Foto não encontrada: {args.imagem}")
            return
        ler_foto(args.imagem, modelo)
    else:
        usar_webcam(args.camera, modelo)


if __name__ == "__main__":
    main()
