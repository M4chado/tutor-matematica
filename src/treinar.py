"""Etapa 2: treina a CNN do zero e registra o aprendizado.

Como rodar (com o .venv ativado, na pasta do projeto):
    python src/treinar.py              # treina as duas versões e gera as figuras
    python src/treinar.py --so-bom     # só o modelo bom
    python src/treinar.py --so-decorando
    python src/treinar.py --figuras    # só refaz as figuras (usa o histórico salvo)

Duas versões são treinadas, para comparação:
  - MODELO BOM: data augmentation + dropout + early stopping + class_weight,
    com todos os dados de treino.
  - MODELO QUE DECORA: sem dropout, sem augmentation, sem early stopping e
    com só 5% dos dados de treino. Ele deve acertar quase 100% no treino e
    bem menos na validação: é o retrato do overfitting.

IMPORTANTE: este script usa só treino e validação. O conjunto de TESTE
nunca é carregado aqui (ele só é usado uma vez, na Etapa 3).
"""
import argparse
import gc
import json
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # esconde mensagens técnicas do TensorFlow

import keras  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # gera as figuras em arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import layers  # noqa: E402

from config import CLASSES, PASTA_DADOS_PROCESSADOS, PASTA_FIGURAS, PASTA_MODELOS, RAIZ, SEMENTE  # noqa: E402
from modelo import criar_modelo  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Hiperparâmetros (definidos no CLAUDE.md)
EPOCAS = 30
TAMANHO_LOTE = 64
PACIENCIA = 5             # early stopping: para se a validação não melhorar em 5 épocas
FRACAO_DECORANDO = 0.05   # o modelo que decora vê só 5% dos dados de treino

ARQUIVO_MODELO_BOM = PASTA_MODELOS / "cnn_simbolos.keras"
ARQUIVO_MODELO_DECORANDO = PASTA_MODELOS / "cnn_decorando.keras"
PASTA_RELATORIOS = RAIZ / "reports"

# Cores dos gráficos (paleta acessível para daltonismo)
COR_TREINO, COR_VALIDACAO = "#2a78d6", "#eb6834"
COR_TEXTO, COR_TEXTO_2, COR_GRADE = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def carregar(nome):
    """Lê treino.npz ou validacao.npz (nunca o teste!)."""
    assert nome in ("treino", "validacao"), "O teste não pode ser usado no treino!"
    dados = np.load(PASTA_DADOS_PROCESSADOS / f"{nome}.npz")
    return dados["X"], dados["y"]


def para_one_hot(y):
    """Transforma o rótulo 5 em [0,0,0,0,0,1,0,...]: o formato da categorical crossentropy."""
    return keras.utils.to_categorical(y, num_classes=len(CLASSES))


def calcular_class_weight(y):
    """Peso de cada classe: classes com poucas imagens (como o ÷) ganham peso maior.

    Fórmula "balanceada": peso = total / (nº de classes × imagens da classe).
    Assim, errar um ÷ "custa" tanto quanto errar vários 1, e a rede não
    aprende a ignorar as classes pequenas.
    """
    contagem = np.bincount(y, minlength=len(CLASSES))
    pesos = len(y) / (len(CLASSES) * contagem)
    return {i: float(p) for i, p in enumerate(pesos)}


def amostra_estratificada(X, y, fracao):
    """Pega a mesma fração de cada classe (pelo menos 1 imagem por classe)."""
    gerador = np.random.default_rng(SEMENTE)
    escolhidos = []
    for c in range(len(CLASSES)):
        indices = np.where(y == c)[0]
        n = max(1, round(len(indices) * fracao))
        escolhidos.extend(gerador.choice(indices, size=n, replace=False))
    escolhidos = gerador.permutation(escolhidos)
    return X[escolhidos], y[escolhidos]


def criar_augmentation():
    """Data augmentation leve: pequenas variações a cada época, para a rede
    não decorar as imagens exatas. NUNCA espelhar: um 2 espelhado deixa de ser 2.

    As variações são pequenas porque a função de pré-processamento já
    centraliza o símbolo. O preenchimento é 1.0 (branco), igual ao fundo.

    interpolation="nearest": cada pixel continua preto OU branco. Com a
    interpolação padrão (bilinear), girar a imagem deixava o traço cinza e
    borrado, um tipo de imagem que a rede nunca vê na validação nem na webcam.
    No primeiro treino isso fez a validação cair de 91% para 62% em 3 épocas.
    """
    preencher = {"fill_mode": "constant", "fill_value": 1.0, "interpolation": "nearest"}
    return keras.Sequential([
        layers.RandomRotation(10 / 360, seed=SEMENTE, **preencher),          # até 10 graus
        layers.RandomTranslation(0.05, 0.05, seed=SEMENTE, **preencher),     # desloca até 5%
        layers.RandomZoom((-0.05, 0.10), seed=SEMENTE, **preencher),         # zoom leve
    ], name="augmentation")


def montar_dataset(X, y, embaralhar, augmentation=None):
    """Prepara os dados em lotes de 64 imagens para o treino.

    Para economizar memória (o notebook tem só 8 GB), as imagens ficam numa
    cópia só, e o que é embaralhado são os NÚMEROS das imagens (índices),
    não as imagens em si.
    """
    imagens = tf.constant(X)
    rotulos = tf.constant(para_one_hot(y))
    ds = tf.data.Dataset.range(len(X))
    if embaralhar:
        ds = ds.shuffle(len(X), seed=SEMENTE, reshuffle_each_iteration=True)
    ds = ds.batch(TAMANHO_LOTE)
    ds = ds.map(lambda idx: (tf.gather(imagens, idx), tf.gather(rotulos, idx)))
    if augmentation is not None:
        ds = ds.map(lambda imgs, rot: (augmentation(imgs, training=True), rot),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Mostrar o progresso
# ---------------------------------------------------------------------------

class Progresso(keras.callbacks.Callback):
    """Mostra uma barra de progresso e um resumo por época, em português.

    No fim de cada época, mede também a acurácia no TREINO do mesmo jeito que
    a validação é medida (sem dropout, sem distorções, sem class_weight).
    É essa medida que vai para o gráfico: só assim treino e validação são
    comparáveis. A acurácia "durante o treino" é mais baixa de propósito,
    porque a rede está com metade dos neurônios desligados e vendo imagens
    distorcidas.
    """

    def __init__(self, total_epocas, dados_treino_limpo, arquivo_historico, bom, n_imagens):
        super().__init__()
        self.total_epocas = total_epocas
        self.dados_treino_limpo = dados_treino_limpo
        self.arquivo_historico = arquivo_historico
        self.bom = bom
        self.n_imagens = n_imagens
        self.inicio = time.time()
        self.historico = {k: [] for k in ("loss", "accuracy", "val_loss", "val_accuracy",
                                          "acc_treino_limpo", "perda_treino_limpo")}

    def salvar_historico(self):
        """Grava as curvas no disco. É chamado A CADA ÉPOCA: se o treino for
        interrompido (ex.: falta de memória), as curvas até ali não se perdem."""
        h = dict(self.historico)
        # Modelo bom: o salvo é o de menor perda na validação. O que decora: o da última época.
        melhor = int(np.argmin(h["val_loss"])) if self.bom else len(h["val_loss"]) - 1
        h["resumo"] = {
            "epocas_treinadas": len(h["loss"]),
            "melhor_epoca": melhor + 1,
            "acuracia_treino": h["acc_treino_limpo"][melhor],
            "acuracia_validacao": h["val_accuracy"][melhor],
            "imagens_treino": self.n_imagens,
            "segundos": time.time() - self.inicio,
        }
        self.arquivo_historico.write_text(json.dumps(h, indent=2), encoding="utf-8")
        return h

    def on_epoch_begin(self, epoca, logs=None):
        self.inicio_epoca = time.time()
        self.epoca = epoca + 1

    def on_train_batch_end(self, lote, logs=None):
        total = self.params["steps"]
        if (lote + 1) % 20 == 0 or lote + 1 == total:
            feito = int(30 * (lote + 1) / total)
            print(f"\r   Época {self.epoca:2d}/{self.total_epocas} [{'█' * feito}{'·' * (30 - feito)}] "
                  f"{100 * (lote + 1) / total:3.0f}%", end="", flush=True)

    def on_epoch_end(self, epoca, logs=None):
        perda, acc = self.model.evaluate(self.dados_treino_limpo, verbose=0)
        self.historico["acc_treino_limpo"].append(float(acc))
        self.historico["perda_treino_limpo"].append(float(perda))
        for chave in ("loss", "accuracy", "val_loss", "val_accuracy"):
            self.historico[chave].append(float(logs[chave]))
        self.salvar_historico()
        segundos = time.time() - self.inicio_epoca
        print(f"\r   Época {self.epoca:2d}/{self.total_epocas} | "
              f"acurácia treino {100 * acc:5.1f}% · validação {100 * logs['val_accuracy']:5.1f}% | "
              f"perda treino {perda:.3f} · validação {logs['val_loss']:.3f} | "
              f"(com distorções: {100 * logs['accuracy']:4.1f}%) | {segundos:.0f}s",
              flush=True)


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def treinar(bom):
    """Treina uma das versões e devolve o histórico (dicionário com as curvas)."""
    keras.utils.set_random_seed(SEMENTE)  # resultados repetíveis
    X_tr, y_tr = carregar("treino")
    X_va, y_va = carregar("validacao")

    if bom:
        titulo, arquivo = "MODELO BOM", ARQUIVO_MODELO_BOM
        augmentation = criar_augmentation()
    else:
        titulo, arquivo = "MODELO QUE DECORA", ARQUIVO_MODELO_DECORANDO
        X_tr, y_tr = amostra_estratificada(X_tr, y_tr, FRACAO_DECORANDO)
        augmentation = None

    print(f"\n=== {titulo} ===")
    print(f"   Imagens de treino: {len(X_tr)} | validação: {len(X_va)}")
    print(f"   Dropout: {'sim' if bom else 'não'} | Augmentation: {'sim' if bom else 'não'} | "
          f"Early stopping: {'sim' if bom else 'não'}")

    modelo = criar_modelo(com_dropout=bom)
    modelo.compile(optimizer=keras.optimizers.Adam(), loss="categorical_crossentropy",
                   metrics=["accuracy"])

    nome_json = "historico_modelo_bom.json" if bom else "historico_modelo_decorando.json"
    progresso = Progresso(EPOCAS, montar_dataset(X_tr, y_tr, embaralhar=False),
                          PASTA_RELATORIOS / nome_json, bom, int(len(X_tr)))
    callbacks = [progresso]
    if bom:
        callbacks += [
            # Para o treino quando a validação parar de melhorar e volta aos melhores pesos
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=PACIENCIA,
                                          restore_best_weights=True),
            # Guarda no disco sempre que a validação bate o recorde
            keras.callbacks.ModelCheckpoint(arquivo, monitor="val_loss", save_best_only=True),
        ]

    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    inicio = time.time()
    modelo.fit(
        montar_dataset(X_tr, y_tr, embaralhar=True, augmentation=augmentation),
        validation_data=montar_dataset(X_va, y_va, embaralhar=False),
        epochs=EPOCAS,
        class_weight=calcular_class_weight(y_tr),
        callbacks=callbacks,
        verbose=0,  # quem mostra o progresso é a classe Progresso
    )
    duracao = time.time() - inicio
    if not bom:
        modelo.save(arquivo)  # sem early stopping: salva o modelo da última época

    h = progresso.salvar_historico()  # versão final do histórico
    r = h["resumo"]
    print(f"\n   Fim! {r['epocas_treinadas']} épocas em {duracao / 60:.1f} min. "
          f"Modelo da época {r['melhor_epoca']}: treino {100 * r['acuracia_treino']:.1f}% · "
          f"validação {100 * r['acuracia_validacao']:.1f}%")
    print(f"   Modelo salvo em {arquivo.relative_to(RAIZ)}")

    # Libera a memória antes do próximo modelo
    del modelo, X_tr, X_va
    keras.backend.clear_session()
    gc.collect()
    return h


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _estilo_eixo(ax):
    ax.grid(axis="y", color=COR_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(COR_TEXTO_2)
    ax.tick_params(colors=COR_TEXTO_2)
    ax.set_xlabel("época", color=COR_TEXTO_2)


def _desenhar_curvas(ax, h, metrica, titulo, marcar_melhor):
    """Desenha treino x validação de uma métrica, com o valor final escrito na ponta."""
    em_porcento = metrica == "accuracy"
    # "treino" = medido sem dropout/distorções, igual à validação (ver classe Progresso)
    chave_treino = "acc_treino_limpo" if em_porcento else "perda_treino_limpo"
    epocas = np.arange(1, len(h[chave_treino]) + 1)
    fator = 100 if em_porcento else 1
    finais = {"treino": h[chave_treino][-1], "validação": h[f"val_{metrica}"][-1]}
    for chave, cor, nome in ((chave_treino, COR_TREINO, "treino"), (f"val_{metrica}", COR_VALIDACAO, "validação")):
        valores = np.array(h[chave]) * fator
        ax.plot(epocas, valores, color=cor, linewidth=2, marker="o", markersize=4, label=nome)
        texto = f"{valores[-1]:.1f}%" if em_porcento else f"{valores[-1]:.2f}"
        # Afasta os dois rótulos: a curva de cima ganha o rótulo um pouco acima, a de baixo, abaixo
        outro = finais["validação" if nome == "treino" else "treino"]
        deslocamento = 7 if finais[nome] >= outro else -7
        ax.annotate(f"{nome}: {texto}", (epocas[-1], valores[-1]), xytext=(6, deslocamento),
                    textcoords="offset points", va="center", fontsize=9, color=COR_TEXTO)
    if marcar_melhor:
        melhor = h["resumo"]["melhor_epoca"]
        ax.axvline(melhor, color=COR_TEXTO_2, linestyle="--", linewidth=1)
        # Posição vertical relativa ao gráfico (0.98 = quase no topo), vale para qualquer escala
        ax.text(melhor, 0.98, "modelo salvo ", fontsize=9, color=COR_TEXTO_2, va="top", ha="right",
                transform=ax.get_xaxis_transform())
    ax.set_title(titulo, loc="left", fontsize=12, fontweight="bold", color=COR_TEXTO)
    ax.set_ylabel("acurácia (%)" if em_porcento else "perda", color=COR_TEXTO_2)
    ax.set_xlim(0.5, len(epocas) + 0.5)
    ax.margins(x=0.15)
    _estilo_eixo(ax)


def figura_um_modelo(h, titulo, arquivo, marcar_melhor):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    _desenhar_curvas(ax1, h, "accuracy", "Acurácia", marcar_melhor)
    _desenhar_curvas(ax2, h, "loss", "Perda (quanto menor, melhor)", marcar_melhor)
    ax1.legend(loc="lower right", frameon=False)
    fig.suptitle(titulo, fontsize=13, fontweight="bold", color=COR_TEXTO, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)
    print(f"   salva: {arquivo.relative_to(RAIZ)}")


def figura_comparacao(h_bom, h_dec, arquivo):
    """As duas versões lado a lado, com a mesma escala, para comparar."""
    fig, eixos = plt.subplots(2, 2, figsize=(13, 8.5), sharey="row")
    r_bom, r_dec = h_bom["resumo"], h_dec["resumo"]
    _desenhar_curvas(eixos[0, 0], h_bom, "accuracy",
                     f"Modelo bom ({r_bom['imagens_treino']} imagens, dropout, augmentation)", True)
    _desenhar_curvas(eixos[0, 1], h_dec, "accuracy",
                     f"Modelo que decora ({r_dec['imagens_treino']} imagens, sem proteções)", False)
    _desenhar_curvas(eixos[1, 0], h_bom, "loss", "Perda: modelo bom", True)
    _desenhar_curvas(eixos[1, 1], h_dec, "loss", "Perda: modelo que decora", False)
    eixos[0, 0].legend(loc="lower right", frameon=False)
    eixos[0, 1].tick_params(labelleft=True)
    eixos[1, 1].tick_params(labelleft=True)
    fig.suptitle("Aprender x decorar: a distância entre treino e validação mostra o overfitting",
                 fontsize=13, fontweight="bold", color=COR_TEXTO, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)
    print(f"   salva: {arquivo.relative_to(RAIZ)}")


def gerar_figuras():
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    historicos = {}
    for nome in ("bom", "decorando"):
        caminho = PASTA_RELATORIOS / f"historico_modelo_{nome}.json"
        if caminho.exists():
            historicos[nome] = json.loads(caminho.read_text(encoding="utf-8"))
    print("\nGerando as figuras...")
    if "bom" in historicos:
        figura_um_modelo(historicos["bom"], "Modelo bom: treino e validação andam juntos",
                         PASTA_FIGURAS / "curvas_modelo_bom.png", marcar_melhor=True)
    if "decorando" in historicos:
        figura_um_modelo(historicos["decorando"], "Modelo que decora: treino dispara, validação fica para trás",
                         PASTA_FIGURAS / "curvas_modelo_decorando.png", marcar_melhor=False)
    if len(historicos) == 2:
        figura_comparacao(historicos["bom"], historicos["decorando"],
                          PASTA_FIGURAS / "comparacao_bom_x_decorando.png")


def main():
    parser = argparse.ArgumentParser(description="Treina a CNN de símbolos matemáticos.")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--so-bom", action="store_true", help="treina só o modelo bom")
    grupo.add_argument("--so-decorando", action="store_true", help="treina só o modelo que decora")
    grupo.add_argument("--figuras", action="store_true", help="só refaz as figuras")
    args = parser.parse_args()

    inicio = time.time()
    if not args.figuras:
        if not args.so_decorando:
            treinar(bom=True)
        if not args.so_bom:
            treinar(bom=False)
    gerar_figuras()
    print(f"\nTempo total: {(time.time() - inicio) / 60:.1f} min")


if __name__ == "__main__":
    main()
