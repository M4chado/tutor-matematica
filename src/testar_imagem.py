"""Etapa 3: classifica fotos de símbolos isolados escritos pelo usuário.

Como rodar (com o .venv ativado, na pasta do projeto):
    python src/testar_imagem.py                     # testa todas as fotos de fotos_teste/
    python src/testar_imagem.py foto1.jpg foto2.png # testa fotos específicas

Para cada foto, mostra os 3 palpites mais prováveis com a confiança e gera
reports/figuras/teste_fotos_usuario.png, com o que a rede "viu" em cada foto.

Caminho de cada foto:
  foto -> igualar iluminação -> isolar o símbolo -> função ÚNICA preprocessar() -> CNN
"""
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from config import CLASSES, PASTA_FIGURAS, PASTA_MODELOS, RAIZ  # noqa: E402
from preprocessamento import normalizar_iluminacao, preprocessar  # noqa: E402
from segmentacao import binarizar_foto  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PASTA_FOTOS_PADRAO = RAIZ / "fotos_teste"
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COR_TEXTO, COR_TEXTO_2 = "#0b0b0b", "#52514e"


def ler_foto(caminho):
    """Lê a foto já na posição certa: o celular guarda a rotação numa
    etiqueta (EXIF) em vez de girar a imagem, e o OpenCV pode ignorá-la."""
    with Image.open(caminho) as imagem:
        imagem = ImageOps.exif_transpose(imagem).convert("L")
        return np.array(imagem)


def isolar_simbolo(limpa):
    """Recorta só o símbolo, descartando sujeira e o que encosta na borda da foto.

    1. Separa tinta e papel (limiar de Otsu).
    2. Encontra os "pedaços" de tinta (componentes conectados).
    3. Descarta pedaços que tocam a borda da foto (mesa, dedo, borda do papel)
       e pedaços minúsculos (poeira), mas mantém partes pequenas de verdade,
       como os pontos do ÷.
    4. Devolve o recorte, com o resto pintado de branco.
    """
    tinta = binarizar_foto(limpa).astype(np.uint8)
    n, rotulos, estatisticas, _ = cv2.connectedComponentsWithStats(tinta, connectivity=8)

    altura, largura = tinta.shape
    candidatos = []
    for i in range(1, n):  # o 0 é o fundo
        x, y, w, h, area = estatisticas[i]
        toca_borda = x == 0 or y == 0 or x + w >= largura or y + h >= altura
        if not toca_borda:
            candidatos.append((i, area))
    if not candidatos:
        return None

    maior = max(area for _, area in candidatos)
    manter = [i for i, area in candidatos if area >= 0.02 * maior]
    mascara = np.isin(rotulos, manter)

    linhas = np.where(mascara.any(axis=1))[0]
    colunas = np.where(mascara.any(axis=0))[0]
    margem = 5
    y0, y1 = max(linhas[0] - margem, 0), min(linhas[-1] + margem + 1, altura)
    x0, x1 = max(colunas[0] - margem, 0), min(colunas[-1] + margem + 1, largura)
    recorte = np.where(mascara, limpa, 255).astype(np.uint8)
    return recorte[y0:y1, x0:x1]


def listar_fotos(argumentos):
    caminhos = [Path(a) for a in argumentos] or [PASTA_FOTOS_PADRAO]
    fotos = []
    for c in caminhos:
        if c.is_dir():
            fotos += sorted(p for p in c.iterdir() if p.suffix.lower() in EXTENSOES)
        elif c.exists():
            fotos.append(c)
        else:
            print(f"⚠️  Não encontrei: {c}")
    return fotos


def figura_resultados(resultados):
    """Uma linha por foto: foto original | recorte | entrada da CNN | 3 palpites."""
    fig, eixos = plt.subplots(len(resultados), 4, figsize=(11, 2.3 * len(resultados)),
                              gridspec_kw={"width_ratios": [1.3, 1, 1, 1.6]}, squeeze=False)
    for linha, r in zip(eixos, resultados):
        for ax in linha:
            ax.set_xticks([]); ax.set_yticks([])
        linha[0].imshow(r["foto"], cmap="gray")
        linha[0].set_title(r["nome"], fontsize=9, color=COR_TEXTO_2)
        if r["entrada"] is not None:
            linha[1].imshow(r["recorte"], cmap="gray")
            linha[2].imshow(r["entrada"][:, :, 0], cmap="gray", vmin=0, vmax=1)
            texto = "\n".join(f"{i + 1}º  {s}   {100 * p:5.1f}%" for i, (s, p) in enumerate(r["palpites"]))
        else:
            texto = "nenhum símbolo\nencontrado"
        linha[1].set_title("recorte", fontsize=9, color=COR_TEXTO_2)
        linha[2].set_title("o que a CNN vê", fontsize=9, color=COR_TEXTO_2)
        linha[3].axis("off")
        linha[3].text(0.05, 0.5, texto, fontsize=14, va="center", family="monospace", color=COR_TEXTO)
    fig.suptitle("Teste com fotos de símbolos escritos à mão", fontsize=12, fontweight="bold", color=COR_TEXTO)
    fig.tight_layout()
    caminho = PASTA_FIGURAS / "teste_fotos_usuario.png"
    fig.savefig(caminho, dpi=110)
    plt.close(fig)
    return caminho


def main():
    fotos = listar_fotos(sys.argv[1:])
    if not fotos:
        print(f"Nenhuma foto encontrada. Coloque as fotos em {PASTA_FOTOS_PADRAO.relative_to(RAIZ)}/ "
              "ou passe o caminho: python src/testar_imagem.py minha_foto.jpg")
        return

    import keras  # carregado só aqui, porque demora alguns segundos

    print("Carregando a CNN...")
    modelo = keras.models.load_model(PASTA_MODELOS / "cnn_simbolos.keras")

    resultados = []
    for caminho in fotos:
        try:
            foto = ler_foto(caminho)
        except Exception as erro:
            print(f"\n{caminho.name}: não consegui abrir ({erro})")
            continue
        limpa = normalizar_iluminacao(foto)
        recorte = isolar_simbolo(limpa)
        r = {"nome": caminho.name, "foto": limpa, "recorte": recorte, "entrada": None, "palpites": []}
        if recorte is None:
            print(f"\n{caminho.name}: nenhum símbolo encontrado (a foto está muito clara ou o símbolo toca a borda?)")
        else:
            entrada = preprocessar(recorte)  # a MESMA função usada no dataset
            probabilidades = modelo.predict(entrada[np.newaxis], verbose=0)[0]
            top3 = np.argsort(probabilidades)[::-1][:3]
            r["entrada"] = entrada
            r["palpites"] = [(CLASSES[i], float(probabilidades[i])) for i in top3]
            print(f"\n{caminho.name}:")
            for posicao, (simbolo, p) in enumerate(r["palpites"], start=1):
                barra = "█" * round(20 * p)
                print(f"   {posicao}º  {simbolo}  {100 * p:5.1f}%  {barra}")
        resultados.append(r)

    if resultados:
        PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
        print(f"\nFigura salva: {figura_resultados(resultados).relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
