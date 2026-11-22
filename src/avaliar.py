"""Etapa 3: avaliação final e honesta no conjunto de TESTE.

Como rodar (com o .venv ativado, na pasta do projeto):
    python src/avaliar.py

Regra do projeto: o teste é usado UMA ÚNICA VEZ. Por isso:
  - Na primeira execução, o modelo classifica o teste e os palpites são
    guardados em data/processed/predicoes_teste.npz, junto com a data e a
    "impressão digital" do modelo em reports/resultado_teste.json.
  - Nas execuções seguintes, o script NÃO passa o teste pela rede de novo:
    ele reaproveita os palpites salvos e só refaz as figuras e o relatório.
  - O teste nunca foi usado para treinar, ajustar nada ou escolher modelo.

O que é gerado:
  - acurácia geral e relatório por classe (no terminal e no JSON)
  - reports/figuras/matriz_confusao.png
  - reports/figuras/erros_exemplos.png
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from config import CLASSES, PASTA_DADOS_PROCESSADOS, PASTA_FIGURAS, PASTA_MODELOS, RAIZ, SEMENTE  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ARQUIVO_MODELO = PASTA_MODELOS / "cnn_simbolos.keras"
ARQUIVO_PREDICOES = PASTA_DADOS_PROCESSADOS / "predicoes_teste.npz"
ARQUIVO_RESULTADO = RAIZ / "reports" / "resultado_teste.json"

COR_TEXTO, COR_TEXTO_2, COR_GRADE = "#0b0b0b", "#52514e", "#e4e3df"
COR_ERRO = "#e34948"


def impressao_digital(caminho):
    """Hash MD5 do arquivo: prova qual modelo exato foi avaliado."""
    return hashlib.md5(caminho.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Palpites no teste (uma única vez)
# ---------------------------------------------------------------------------

def obter_predicoes(reavaliar):
    """Devolve as probabilidades que o modelo deu para cada imagem de teste."""
    if ARQUIVO_PREDICOES.exists() and not reavaliar:
        salvo = np.load(ARQUIVO_PREDICOES)
        print("O teste JÁ foi avaliado antes. Reaproveitando os palpites salvos")
        print(f"(o teste não passa pela rede de novo). Avaliado em: {str(salvo['data'])}\n")
        return salvo["probabilidades"], False

    import keras  # só carrega o TensorFlow se precisar mesmo avaliar

    print("Avaliando o modelo no conjunto de TESTE (uso único)...")
    modelo = keras.models.load_model(ARQUIVO_MODELO)
    teste = np.load(PASTA_DADOS_PROCESSADOS / "teste.npz")
    probabilidades = modelo.predict(teste["X"], batch_size=256, verbose=0)
    agora = datetime.datetime.now().isoformat(timespec="seconds")
    np.savez_compressed(ARQUIVO_PREDICOES, probabilidades=probabilidades, data=agora)
    print(f"Palpites guardados em {ARQUIVO_PREDICOES.relative_to(RAIZ)}\n")
    return probabilidades, True


# ---------------------------------------------------------------------------
# 2. Métricas
# ---------------------------------------------------------------------------

def matriz_confusao(y_real, y_previsto):
    """Linha = símbolo verdadeiro, coluna = palpite da rede."""
    n = len(CLASSES)
    matriz = np.zeros((n, n), dtype=int)
    np.add.at(matriz, (y_real, y_previsto), 1)
    return matriz


def relatorio_por_classe(matriz):
    """Para cada classe:
    - taxa de acerto (recall): dos símbolos que ERAM essa classe, quantos a rede acertou
    - precisão: quando a rede DISSE essa classe, quantas vezes estava certa
    - F1: média que equilibra as duas
    """
    acertos = np.diag(matriz)
    total_real = matriz.sum(axis=1)
    total_previsto = matriz.sum(axis=0)
    recall = acertos / np.maximum(total_real, 1)
    precisao = acertos / np.maximum(total_previsto, 1)
    f1 = 2 * precisao * recall / np.maximum(precisao + recall, 1e-9)
    return [
        {"classe": c, "imagens": int(total_real[i]), "acertos": int(acertos[i]),
         "taxa_acerto": float(recall[i]), "precisao": float(precisao[i]), "f1": float(f1[i])}
        for i, c in enumerate(CLASSES)
    ]


def maiores_confusoes(matriz, quantas=8):
    """As trocas mais frequentes (fora da diagonal)."""
    fora = matriz.copy()
    np.fill_diagonal(fora, 0)
    pares = np.dstack(np.unravel_index(np.argsort(fora, axis=None)[::-1], fora.shape))[0]
    resultado = []
    for real, prev in pares[:quantas]:
        if fora[real, prev] == 0:
            break
        resultado.append({"real": CLASSES[real], "previsto": CLASSES[prev], "vezes": int(fora[real, prev]),
                          "porcento_da_classe": float(fora[real, prev] / matriz[real].sum())})
    return resultado


# ---------------------------------------------------------------------------
# 3. Figuras
# ---------------------------------------------------------------------------

def figura_matriz(matriz, acuracia):
    """Matriz de confusão em porcentagem de cada linha (cada símbolo verdadeiro).

    Usar porcentagem é importante porque as classes têm tamanhos diferentes
    (o ÷ tem 23 imagens de teste e o 1 tem 450).
    """
    porcento = 100 * matriz / np.maximum(matriz.sum(axis=1, keepdims=True), 1)
    n = len(CLASSES)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    imagem = ax.imshow(porcento, cmap="Blues", vmin=0, vmax=100)
    for i in range(n):
        for j in range(n):
            if matriz[i, j] == 0:
                continue
            valor = porcento[i, j]
            texto = f"{valor:.0f}" if valor >= 1 else "<1"
            cor = "white" if valor > 55 else COR_TEXTO
            ax.text(j, i, texto, ha="center", va="center", fontsize=8.5, color=cor)
    ax.set_xticks(range(n), CLASSES, fontsize=13)
    ax.set_yticks(range(n), [f"{c}  ({matriz[i].sum()})" for i, c in enumerate(CLASSES)], fontsize=11)
    ax.set_xlabel("palpite da rede", fontsize=12, color=COR_TEXTO_2)
    ax.set_ylabel("símbolo verdadeiro  (nº de imagens de teste)", fontsize=12, color=COR_TEXTO_2)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    for lado in ax.spines.values():
        lado.set_visible(False)
    barra = fig.colorbar(imagem, ax=ax, fraction=0.04, pad=0.02)
    barra.set_label("% das imagens daquele símbolo", color=COR_TEXTO_2)
    barra.outline.set_visible(False)
    fig.suptitle(f"Matriz de confusão no conjunto de teste — acurácia geral {100 * acuracia:.1f}%",
                 fontsize=13, fontweight="bold", color=COR_TEXTO, y=0.02, va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    caminho = PASTA_FIGURAS / "matriz_confusao.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def figura_erros(X, y_real, probabilidades, quantos=32):
    """Exemplos de imagens que a rede errou, com o palpite e a confiança."""
    y_prev = probabilidades.argmax(axis=1)
    erros = np.where(y_prev != y_real)[0]
    gerador = np.random.default_rng(SEMENTE)
    escolhidos = gerador.choice(erros, size=min(quantos, len(erros)), replace=False)
    escolhidos = escolhidos[np.argsort(y_real[escolhidos], kind="stable")]  # agrupa pelo símbolo verdadeiro

    colunas = 8
    linhas = int(np.ceil(len(escolhidos) / colunas))
    fig, eixos = plt.subplots(linhas, colunas, figsize=(colunas * 1.55, linhas * 2.2))
    for k, ax in enumerate(np.ravel(eixos)):
        ax.set_xticks([]); ax.set_yticks([])
        if k >= len(escolhidos):
            ax.axis("off")
            continue
        i = escolhidos[k]
        ax.imshow(X[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        confianca = probabilidades[i, y_prev[i]]
        ax.set_title(f"era {CLASSES[y_real[i]]}\nrede: {CLASSES[y_prev[i]]} ({100 * confianca:.0f}%)",
                     fontsize=10, color=COR_TEXTO)
        for lado in ax.spines.values():
            lado.set_color(COR_ERRO)
            lado.set_linewidth(1.5)
    fig.suptitle(f"Exemplos de erros no teste ({len(erros)} erros em {len(y_real)} imagens)",
                 fontsize=12, fontweight="bold", color=COR_TEXTO)
    fig.tight_layout()
    caminho = PASTA_FIGURAS / "erros_exemplos.png"
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    return caminho


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Avaliação final no conjunto de teste.")
    parser.add_argument("--reavaliar", action="store_true",
                        help="força passar o teste pela rede de novo (só se o modelo foi retreinado "
                             "por um motivo que NÃO veio do teste)")
    args = parser.parse_args()

    teste = np.load(PASTA_DADOS_PROCESSADOS / "teste.npz")
    X, y_real = teste["X"], teste["y"]
    probabilidades, avaliou_agora = obter_predicoes(args.reavaliar)
    y_prev = probabilidades.argmax(axis=1)

    matriz = matriz_confusao(y_real, y_prev)
    acuracia = float((y_prev == y_real).mean())
    por_classe = relatorio_por_classe(matriz)
    media_por_classe = float(np.mean([c["taxa_acerto"] for c in por_classe]))
    confusoes = maiores_confusoes(matriz)

    print(f"ACURÁCIA GERAL NO TESTE: {100 * acuracia:.1f}%  ({int((y_prev == y_real).sum())} de {len(y_real)})")
    print(f"Média da taxa de acerto por classe: {100 * media_por_classe:.1f}%  "
          "(cada símbolo conta igual, mesmo os que têm poucas imagens)\n")
    print(f"{'classe':>6} {'imagens':>8} {'acertos':>8} {'taxa de acerto':>15} {'precisão':>9} {'F1':>6}")
    for c in por_classe:
        print(f"{c['classe']:>6} {c['imagens']:>8} {c['acertos']:>8} {100 * c['taxa_acerto']:>14.1f}% "
              f"{100 * c['precisao']:>8.1f}% {c['f1']:>6.2f}")
    print("\nTrocas mais frequentes:")
    for c in confusoes:
        print(f"   {c['real']} lido como {c['previsto']}: {c['vezes']} vezes "
              f"({100 * c['porcento_da_classe']:.1f}% dos {c['real']})")

    # Registro permanente do resultado (vai para o Git)
    salvo = np.load(ARQUIVO_PREDICOES)
    resultado = {
        "avaliado_em": str(salvo["data"]),
        "modelo": str(ARQUIVO_MODELO.relative_to(RAIZ)),
        "modelo_md5": impressao_digital(ARQUIVO_MODELO),
        "imagens_teste": int(len(y_real)),
        "acuracia": acuracia,
        "media_taxa_acerto_por_classe": media_por_classe,
        "por_classe": por_classe,
        "maiores_confusoes": confusoes,
        "matriz_confusao": matriz.tolist(),
    }
    if not avaliou_agora and ARQUIVO_RESULTADO.exists():
        anterior = json.loads(ARQUIVO_RESULTADO.read_text(encoding="utf-8"))
        if anterior.get("modelo_md5") != resultado["modelo_md5"]:
            print("\n⚠️  Atenção: o modelo atual é diferente do que foi avaliado no teste!")
        resultado["modelo_md5"] = anterior.get("modelo_md5", resultado["modelo_md5"])
    ARQUIVO_RESULTADO.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nFiguras:")
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    for caminho in (figura_matriz(matriz, acuracia), figura_erros(X, y_real, probabilidades)):
        print(f"   salva: {caminho.relative_to(RAIZ)}")
    print(f"   resultado registrado em: {ARQUIVO_RESULTADO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
