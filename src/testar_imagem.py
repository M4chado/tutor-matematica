"""Etapa 3: classifica fotos de símbolos isolados escritos pelo usuário.

Como rodar (com o .venv ativado, na pasta do projeto):
    python src/testar_imagem.py                     # testa todas as fotos de fotos_teste/
    python src/testar_imagem.py foto1.jpg foto2.png # testa fotos específicas

Para cada símbolo encontrado, mostra os 3 palpites mais prováveis com a
confiança e gera reports/figuras/teste_fotos_usuario.png, com o que a rede "viu".

Caminho de cada foto (o mesmo da webcam):
  foto -> achar a folha -> igualar iluminação -> recortar cada símbolo cercado de papel
       -> função ÚNICA preprocessar() -> CNN
Mão, roupa e o que estiver atrás da folha são ignorados.
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

from config import PASTA_FIGURAS, PASTA_MODELOS, RAIZ  # noqa: E402
from segmentacao import classificar, desenhar_leitura, segmentar  # noqa: E402

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
        rgb = np.array(ImageOps.exif_transpose(imagem).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # colorida: a cor ajuda a achar a folha


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
    """Uma linha por símbolo: foto (com a caixa) | recorte | entrada da CNN | 3 palpites."""
    fig, eixos = plt.subplots(len(resultados), 4, figsize=(11, 2.3 * len(resultados)),
                              gridspec_kw={"width_ratios": [1.3, 1, 1, 1.6]}, squeeze=False)
    for linha, r in zip(eixos, resultados):
        for ax in linha:
            ax.set_xticks([]); ax.set_yticks([])
        linha[0].imshow(cv2.cvtColor(r["foto"], cv2.COLOR_BGR2RGB))
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
        # O mesmo caminho da webcam: só a tinta cercada de papel, pela função única preprocessar()
        # uma_linha=False: aqui os símbolos podem estar soltos pela folha (não é uma conta)
        simbolos = classificar(segmentar(foto, so_na_folha=True, uma_linha=False), modelo)
        if not simbolos:
            print(f"\n{caminho.name}: nenhum símbolo encontrado "
                  "(o traço está fino/claro demais ou o símbolo encosta na borda da folha?)")
            resultados.append({"nome": caminho.name, "foto": foto, "entrada": None})
            continue
        for n, s in enumerate(simbolos, start=1):
            nome = caminho.name if len(simbolos) == 1 else f"{caminho.name} ({n} de {len(simbolos)})"
            print(f"\n{nome}:")
            for posicao, (simbolo, p) in enumerate(s["top3"], start=1):
                print(f"   {posicao}º  {simbolo}  {100 * p:5.1f}%  {'█' * round(20 * p)}")
            resultados.append({"nome": nome, "foto": desenhar_leitura(foto, [s]), "recorte": s["recorte"],
                               "entrada": s["entrada"], "palpites": s["top3"]})

    if resultados:
        PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
        print(f"\nFigura salva: {figura_resultados(resultados).relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
