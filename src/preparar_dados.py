"""Etapa 1: transforma o dataset bruto em dados limpos, balanceados e divididos.

Como rodar (com o .venv ativado, na pasta do projeto):
    python src/preparar_dados.py

O que ele faz:
  1. Para cada uma das 17 classes, lista as imagens da pasta correspondente.
  2. Remove imagens repetidas (arquivos idênticos), para que a mesma imagem
     não caia no treino E no teste, o que deixaria o teste "fácil" demais.
  3. Sorteia no máximo 3.000 imagens por classe (equilíbrio).
  4. Aplica a função única de pré-processamento em cada imagem.
  5. Divide cada classe em treino (70%), validação (15%) e teste (15%).
  6. Salva tudo em data/processed/ e gera as figuras em reports/figuras/.
"""
import hashlib
import sys
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # gera as figuras direto em arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import (  # noqa: E402
    CLASSES, LIMITE_POR_CLASSE, MAPA_PASTAS, PASTA_DADOS_PROCESSADOS, PASTA_FIGURAS,
    PASTA_IMAGENS_BRUTAS, PROPORCAO_TREINO, PROPORCAO_VALIDACAO, SEMENTE,
)
from preprocessamento import carregar_imagem, preprocessar  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")  # acentos e símbolos no terminal do Windows
except Exception:
    pass

# Cores do gráfico (paleta acessível para daltonismo)
COR_TREINO, COR_VALIDACAO, COR_TESTE = "#2a78d6", "#eb6834", "#1baf7a"
COR_TEXTO, COR_TEXTO_2, COR_GRADE = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------------------
# 1 e 2. Listar as imagens e remover repetidas
# ---------------------------------------------------------------------------

def listar_imagens_sem_repeticao():
    """Devolve {classe: [caminhos]} sem arquivos repetidos.

    Uma "impressão digital" (hash MD5) de cada arquivo mostra se dois arquivos
    são idênticos. Repetidos dentro da mesma classe: fica só um.
    Iguais em classes DIFERENTES: são descartados, porque o rótulo é duvidoso.
    """
    dono_do_hash = {}                  # hash -> classe onde apareceu
    hashes_conflitantes = set()        # hashes que aparecem em 2 classes
    caminhos_por_hash = defaultdict(dict)
    total_por_classe = {}

    for classe in CLASSES:
        pasta = PASTA_IMAGENS_BRUTAS / MAPA_PASTAS[classe]
        arquivos = sorted(p for p in pasta.iterdir() if p.suffix.lower() in (".jpg", ".png"))
        total_por_classe[classe] = len(arquivos)
        for caminho in arquivos:
            h = hashlib.md5(caminho.read_bytes()).hexdigest()
            if h in dono_do_hash and dono_do_hash[h] != classe:
                hashes_conflitantes.add(h)
            dono_do_hash.setdefault(h, classe)
            # setdefault: se o hash já existe nesta classe, mantém o primeiro arquivo
            caminhos_por_hash[classe].setdefault(h, caminho)

    unicos = {}
    for classe in CLASSES:
        unicos[classe] = sorted(
            c for h, c in caminhos_por_hash[classe].items() if h not in hashes_conflitantes
        )
    return total_por_classe, unicos, len(hashes_conflitantes)


# ---------------------------------------------------------------------------
# 3 a 5. Sortear, pré-processar e dividir
# ---------------------------------------------------------------------------

def sortear_e_dividir(unicos):
    """Sorteia até 3.000 imagens por classe e divide em treino/validação/teste.

    A divisão é feita DENTRO de cada classe (estratificada): assim, as três
    partes ficam com a mesma proporção de cada símbolo.
    """
    gerador = np.random.default_rng(SEMENTE)
    partes = {"treino": [], "validacao": [], "teste": []}  # listas de (caminho, rótulo)

    for rotulo, classe in enumerate(CLASSES):
        caminhos = list(unicos[classe])
        gerador.shuffle(caminhos)
        caminhos = caminhos[:LIMITE_POR_CLASSE]

        n = len(caminhos)
        n_treino = round(n * PROPORCAO_TREINO)
        n_validacao = round(n * PROPORCAO_VALIDACAO)
        partes["treino"] += [(c, rotulo) for c in caminhos[:n_treino]]
        partes["validacao"] += [(c, rotulo) for c in caminhos[n_treino:n_treino + n_validacao]]
        partes["teste"] += [(c, rotulo) for c in caminhos[n_treino + n_validacao:]]

    # Embaralha cada parte para as classes não ficarem em sequência
    for nome in partes:
        ordem = gerador.permutation(len(partes[nome]))
        partes[nome] = [partes[nome][i] for i in ordem]
    return partes


def processar_parte(itens, nome):
    """Aplica a função única de pré-processamento em todas as imagens de uma parte."""
    X = np.empty((len(itens), 45, 45, 1), dtype=np.float32)
    y = np.empty(len(itens), dtype=np.int64)
    for i, (caminho, rotulo) in enumerate(itens):
        X[i] = preprocessar(carregar_imagem(caminho))
        y[i] = rotulo
        if (i + 1) % 5000 == 0 or i + 1 == len(itens):
            print(f"   {nome}: {i + 1}/{len(itens)} imagens processadas")
    # Caminho relativo de cada imagem, para rastrear de onde ela veio (ex.: "times/exp123.jpg")
    arquivos = np.array([f"{c.parent.name}/{c.name}" for c, _ in itens])
    return X, y, arquivos


# ---------------------------------------------------------------------------
# 6. Figuras
# ---------------------------------------------------------------------------

def figura_contagem(total_por_classe, contagem_partes):
    """Dois gráficos lado a lado: quantas imagens existem e quantas usamos."""
    plt.rcParams.update({"font.size": 11, "text.color": COR_TEXTO, "axes.labelcolor": COR_TEXTO_2,
                         "xtick.color": COR_TEXTO_2, "ytick.color": COR_TEXTO_2})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    posicoes = np.arange(len(CLASSES))

    # Esquerda: imagens disponíveis no dataset (muito desequilibrado)
    totais = [total_por_classe[c] for c in CLASSES]
    ax1.bar(posicoes, totais, width=0.75, color=COR_TREINO)
    ax1.set_title("Imagens disponíveis no dataset", loc="left", fontsize=13, fontweight="bold")
    ax1.set_ylabel("número de imagens")

    # Direita: imagens usadas, divididas em treino/validação/teste (empilhadas)
    tr = np.array([contagem_partes["treino"][c] for c in CLASSES])
    va = np.array([contagem_partes["validacao"][c] for c in CLASSES])
    te = np.array([contagem_partes["teste"][c] for c in CLASSES])
    borda = {"edgecolor": "white", "linewidth": 1.5}  # pequeno espaço entre as partes
    ax2.bar(posicoes, tr, width=0.75, color=COR_TREINO, label="treino (70%)", **borda)
    ax2.bar(posicoes, va, width=0.75, bottom=tr, color=COR_VALIDACAO, label="validação (15%)", **borda)
    ax2.bar(posicoes, te, width=0.75, bottom=tr + va, color=COR_TESTE, label="teste (15%)", **borda)
    ax2.axhline(LIMITE_POR_CLASSE, color=COR_TEXTO_2, linestyle="--", linewidth=1)
    ax2.text(len(CLASSES) - 0.5, LIMITE_POR_CLASSE * 1.02, f"limite: {LIMITE_POR_CLASSE}",
             ha="right", va="bottom", color=COR_TEXTO_2, fontsize=10)
    # Rótulo só nas classes que ficaram abaixo do limite (as que merecem atenção)
    for p, total in zip(posicoes, tr + va + te):
        if total < LIMITE_POR_CLASSE:
            ax2.text(p, total + 40, str(total), ha="center", va="bottom", fontsize=10, color=COR_TEXTO)
    ax2.set_title("Imagens usadas, após o limite por classe", loc="left", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, LIMITE_POR_CLASSE * 1.15)
    ax2.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=3, frameon=False)

    for ax in (ax1, ax2):
        ax.set_xticks(posicoes, CLASSES, fontsize=13, color=COR_TEXTO)
        ax.grid(axis="y", color=COR_GRADE, linewidth=0.8)
        ax.set_axisbelow(True)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color(COR_TEXTO_2)
        ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    caminho = PASTA_FIGURAS / "contagem_por_classe.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def figura_exemplos(X, y, por_classe=10):
    """Grade com exemplos de cada classe, já pré-processados (o que a CNN 'vê')."""
    gerador = np.random.default_rng(SEMENTE)
    fig, eixos = plt.subplots(len(CLASSES), por_classe, figsize=(por_classe * 0.9, len(CLASSES) * 0.95))
    for rotulo, classe in enumerate(CLASSES):
        indices = np.where(y == rotulo)[0]
        escolhidos = gerador.choice(indices, size=min(por_classe, len(indices)), replace=False)
        for j in range(por_classe):
            ax = eixos[rotulo, j]
            ax.set_xticks([]); ax.set_yticks([])
            for lado in ax.spines.values():
                lado.set_color(COR_GRADE)
            if j < len(escolhidos):
                ax.imshow(X[escolhidos[j], :, :, 0], cmap="gray", vmin=0, vmax=1)
            if j == 0:
                ax.set_ylabel(classe, rotation=0, fontsize=16, labelpad=18, va="center", color=COR_TEXTO)
    fig.suptitle("Exemplos de cada classe após o pré-processamento (conjunto de treino)",
                 fontsize=12, fontweight="bold", color=COR_TEXTO)
    fig.tight_layout()
    caminho = PASTA_FIGURAS / "exemplos_por_classe.png"
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def figura_antes_depois(itens_treino):
    """Mostra o efeito do pré-processamento: original x processada,
    incluindo um exemplo que imita a webcam (traço grosso, papel cinza)."""
    import cv2

    exemplos = []
    vistos = set()
    for caminho, rotulo in itens_treino:  # um exemplo de algumas classes
        if rotulo not in vistos and CLASSES[rotulo] in ("2", "8", "+", "×", "÷", "=", "x"):
            vistos.add(rotulo)
            exemplos.append((CLASSES[rotulo], carregar_imagem(caminho)))

    # Imitação de foto da webcam: "7" com caneta grossa em papel cinza
    webcam = np.full((300, 240), 190, np.uint8)
    cv2.line(webcam, (40, 50), (200, 50), 30, 18)
    cv2.line(webcam, (200, 50), (90, 270), 30, 18)
    exemplos.append(("7 (webcam)", webcam))

    fig, eixos = plt.subplots(2, len(exemplos), figsize=(len(exemplos) * 1.5, 3.6))
    for j, (nome, original) in enumerate(exemplos):
        eixos[0, j].imshow(original, cmap="gray", vmin=0, vmax=255)
        eixos[1, j].imshow(preprocessar(original)[:, :, 0], cmap="gray", vmin=0, vmax=1)
        eixos[0, j].set_title(nome, fontsize=11, color=COR_TEXTO)
        for i in range(2):
            eixos[i, j].set_xticks([]); eixos[i, j].set_yticks([])
    eixos[0, 0].set_ylabel("original", fontsize=10, color=COR_TEXTO_2)
    eixos[1, 0].set_ylabel("processada", fontsize=10, color=COR_TEXTO_2)
    fig.suptitle("Função única de pré-processamento: dataset e webcam ficam no mesmo formato",
                 fontsize=11, fontweight="bold", color=COR_TEXTO)
    fig.tight_layout()
    caminho = PASTA_FIGURAS / "preprocessamento_antes_depois.png"
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    return caminho


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    inicio = time.time()
    PASTA_DADOS_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

    print("1) Lendo as pastas e removendo imagens repetidas...")
    total_por_classe, unicos, n_conflitos = listar_imagens_sem_repeticao()

    print("2) Sorteando até 3.000 por classe e dividindo 70/15/15...")
    partes = sortear_e_dividir(unicos)

    # Contagem por classe em cada parte
    contagem = {nome: {c: 0 for c in CLASSES} for nome in partes}
    for nome, itens in partes.items():
        for _, rotulo in itens:
            contagem[nome][CLASSES[rotulo]] += 1

    # Tabela de conferência
    print(f"\n{'classe':>6} {'pasta':>7} {'no dataset':>11} {'repetidas':>10} "
          f"{'usadas':>7} {'treino':>7} {'valid.':>7} {'teste':>6}")
    for classe in CLASSES:
        repetidas = total_por_classe[classe] - len(unicos[classe])
        usadas = sum(contagem[n][classe] for n in partes)
        print(f"{classe:>6} {MAPA_PASTAS[classe]:>7} {total_por_classe[classe]:>11} {repetidas:>10} "
              f"{usadas:>7} {contagem['treino'][classe]:>7} {contagem['validacao'][classe]:>7} "
              f"{contagem['teste'][classe]:>6}")
    totais = {n: len(itens) for n, itens in partes.items()}
    print(f"{'TOTAL':>6} {'':>7} {sum(total_por_classe.values()):>11} "
          f"{sum(total_por_classe.values()) - sum(len(u) for u in unicos.values()):>10} "
          f"{sum(totais.values()):>7} {totais['treino']:>7} {totais['validacao']:>7} {totais['teste']:>6}")
    print(f"\n   Imagens idênticas em classes diferentes (descartadas): {n_conflitos}")

    # Garantia: nenhuma imagem aparece em duas partes ao mesmo tempo
    conjuntos = {n: {c for c, _ in itens} for n, itens in partes.items()}
    sobreposicao = (len(conjuntos["treino"] & conjuntos["validacao"])
                    + len(conjuntos["treino"] & conjuntos["teste"])
                    + len(conjuntos["validacao"] & conjuntos["teste"]))
    print(f"   Imagens repetidas entre treino/validação/teste: {sobreposicao}")
    assert sobreposicao == 0, "Erro grave: há imagens em mais de uma parte!"

    print("\n3) Aplicando a função única de pré-processamento...")
    dados = {}
    for nome, itens in partes.items():
        X, y, arquivos = processar_parte(itens, nome)
        dados[nome] = (X, y)
        destino = PASTA_DADOS_PROCESSADOS / f"{nome}.npz"
        # savez_compressed: arquivo bem menor (as imagens têm muito branco)
        np.savez_compressed(destino, X=X, y=y, arquivos=arquivos, classes=np.array(CLASSES))
        print(f"   salvo: {destino.relative_to(PASTA_DADOS_PROCESSADOS.parent.parent)} "
              f"({destino.stat().st_size / 1e6:.1f} MB)")

    print("\n4) Gerando as figuras...")
    for caminho in (
        figura_contagem(total_por_classe, contagem),
        figura_exemplos(*dados["treino"]),
        figura_antes_depois(partes["treino"]),
    ):
        print(f"   salva: {caminho.relative_to(PASTA_FIGURAS.parent.parent)}")

    print(f"\nPronto! Tempo total: {time.time() - inicio:.0f} s")


if __name__ == "__main__":
    main()
