"""Etapa 6: escolher o LIMITE_CONFIANCA (abaixo dele, o agente pergunta ao visitante).

Usamos o conjunto de VALIDAÇÃO (nunca o de teste, que só foi usado uma vez na
Etapa 3). Para cada limite possível, simulamos a regra do agente
(agente.opcoes_da_duvida) em todos os símbolos e medimos:

  - quantas vezes o agente pergunta (perguntar demais irrita o visitante);
  - quantos erros da CNN passam sem pergunta (o agente calcularia a conta errada).

Erros entre x e × não contam como "passando", porque o contexto resolve
(resolver_ambiguidade_x, Etapa 5).

Rodar: python src/calibrar_confianca.py
"""
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agente import opcoes_da_duvida
from config import CLASSES, LIMITE_CONFIANCA, PASTA_DADOS_PROCESSADOS, PASTA_FIGURAS, PASTA_MODELOS, RAIZ
from ferramentas import CRUZES

LIMITES = np.round(np.arange(0.30, 0.996, 0.01), 2)
SIMBOLOS_POR_CONTA = 6     # uma conta típica, como 12+7×3, tem uns 6 símbolos
COR_PERGUNTA, COR_ERRO = "#2a78d6", "#eb6834"
COR_TEXTO, COR_TEXTO_2, COR_GRADE = "#0b0b0b", "#52514e", "#e4e3df"


def probabilidades_validacao():
    import keras
    dados = np.load(PASTA_DADOS_PROCESSADOS / "validacao.npz")
    modelo = keras.models.load_model(PASTA_MODELOS / "cnn_simbolos.keras")
    return modelo.predict(dados["X"], batch_size=256, verbose=0), dados["y"]


def simular(probabilidades, y, limite):
    """Aplica a regra do agente a cada símbolo da validação com este limite."""
    perguntas = erros_passando = 0
    for p, real in zip(probabilidades, y):
        ordem = np.argsort(p)[::-1][:3]
        simbolo = {"simbolo": CLASSES[ordem[0]], "confianca": float(p[ordem[0]]),
                   "top3": [(CLASSES[i], float(p[i])) for i in ordem]}
        if opcoes_da_duvida(simbolo, limite):
            perguntas += 1
            continue
        palpite, certo = simbolo["simbolo"], CLASSES[real]
        if palpite != certo and not (palpite in CRUZES and certo in CRUZES):
            erros_passando += 1
    n = len(y)
    return perguntas / n, erros_passando / n


def figura(tabela, escolhido, arquivo):
    limites = [l["limite"] for l in tabela]
    pergunta = [100 * l["perguntas"] for l in tabela]
    erro = [100 * l["erros_passando"] for l in tabela]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(limites, pergunta, color=COR_PERGUNTA, linewidth=2, label="símbolos em que o agente pergunta")
    ax.plot(limites, erro, color=COR_ERRO, linewidth=2, label="erros da CNN que passam sem pergunta")
    ax.axvline(escolhido, color=COR_TEXTO_2, linestyle="--", linewidth=1)
    linha = next(l for l in tabela if l["limite"] == escolhido)
    ax.text(escolhido, 0.98, f" limite escolhido: {escolhido:.2f}", transform=ax.get_xaxis_transform(),
            va="top", ha="left", fontsize=9, color=COR_TEXTO_2)
    for valor, cor in ((100 * linha["perguntas"], COR_PERGUNTA), (100 * linha["erros_passando"], COR_ERRO)):
        ax.plot([escolhido], [valor], "o", color=cor, markersize=8, markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"{valor:.1f}%", (escolhido, valor), xytext=(8, 6), textcoords="offset points",
                    fontsize=9, color=COR_TEXTO)
    ax.set_xlabel("limite de confiança (abaixo dele, o agente pergunta)", color=COR_TEXTO_2)
    ax.set_ylabel("% dos símbolos da validação", color=COR_TEXTO_2)
    ax.set_title("Calibração do limite de confiança (conjunto de validação)", loc="left",
                 fontsize=12, fontweight="bold", color=COR_TEXTO)
    ax.grid(axis="y", color=COR_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(COR_TEXTO_2)
    ax.tick_params(colors=COR_TEXTO_2)
    ax.legend(frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("Calculando as probabilidades da CNN no conjunto de VALIDAÇÃO...")
    probabilidades, y = probabilidades_validacao()
    tabela = []
    for limite in LIMITES:
        perguntas, erros = simular(probabilidades, y, float(limite))
        conta_com_pergunta = 1 - (1 - perguntas) ** SIMBOLOS_POR_CONTA
        tabela.append({"limite": float(limite), "perguntas": perguntas, "erros_passando": erros,
                       "contas_com_pergunta": conta_com_pergunta})

    sem_regra = simular(probabilidades, y, 0.0)[1]
    print(f"\nSem perguntar nunca, {100 * sem_regra:.1f}% dos símbolos sairiam errados (fora a troca x/×).\n")
    print(" limite | pergunta em | erros que passam | contas de 6 símbolos com pergunta")
    for l in tabela:
        if round(l["limite"] * 100) % 5 == 0:
            print(f"  {l['limite']:.2f}  |   {100 * l['perguntas']:5.1f}%    |      {100 * l['erros_passando']:4.1f}%      |"
                  f"   {100 * l['contas_com_pergunta']:4.1f}%")

    escolhido = float(sys.argv[1]) if len(sys.argv) > 1 else LIMITE_CONFIANCA
    figura(tabela, escolhido, PASTA_FIGURAS / "calibracao_confianca.png")
    with open(RAIZ / "reports" / "calibracao_confianca.json", "w", encoding="utf-8") as f:
        json.dump({"conjunto": "validacao", "sem_perguntar": sem_regra, "escolhido": escolhido, "tabela": tabela},
                  f, ensure_ascii=False, indent=1)
    print(f"\nFigura: reports/figuras/calibracao_confianca.png (limite marcado: {escolhido:.2f})")


if __name__ == "__main__":
    main()
