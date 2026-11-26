"""Gera o banner do 5º Workshop de IA (UniRV) preenchido com o CalcuLetra.

Rodar (com o .venv ativado, na pasta do projeto):
    python banner/gerar_banner.py

Lê o modelo "Banner Workshop IA.png" (na pasta do projeto) e cria, em banner/:
  - figura_metodologia.png   diagrama de como o sistema funciona
  - figura_resultados.png    aprende x decora + exemplo de leitura
  - banner_preenchido.png    o banner completo (nome e e-mail ficam para você)
Os textos ficam no topo deste arquivo, para editar fácil.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parent
sys.path.insert(0, str(RAIZ / "src"))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

# ---------------------------------------------------------------------------
# Textos (edite aqui)
# ---------------------------------------------------------------------------
TITULO = ["CalcuLetra: Tutor de Matemática que Lê sua Letra",
          "com Rede Neural e Agente de IA Local"]

AUTORES = [  # (nome, e-mail), na ordem em que aparecem no banner (da esquerda para a direita)
    ("Eduardo Machado Martins", "eduardo.martins@academico.unirv.edu.br"),
    ("Augusto Wobeto Carvalho", "augusto.carvalho@academico.unirv.edu.br"),
]

INTRODUCAO = [
    "Aprender matemática envolve escrever contas à mão, mas a maioria das ferramentas digitais "
    "exige que o aluno digite a expressão. Além disso, os modelos de linguagem (LLMs) explicam bem, "
    "mas podem errar cálculos, pois geram texto prevendo palavras, e não calculando.",
    "Este trabalho apresenta o CalcuLetra, um tutor que lê contas escritas à mão por uma webcam e as "
    "resolve e explica passo a passo, em português. O sistema une duas áreas da Inteligência "
    "Artificial: uma rede neural convolucional (CNN), treinada do zero, reconhece os símbolos; e um "
    "agente com LLM local decide o que fazer, usa ferramentas de cálculo exato e confere a própria "
    "resposta. O objetivo é mostrar que a rede realmente aprende, e não apenas decora, e que um "
    "agente bem projetado evita respostas erradas, funcionando inteiramente sem internet.",
]

METODOLOGIA = [
    "Foram usadas 34.722 imagens únicas do dataset Handwritten Math Symbols (Kaggle), após a "
    "remoção de 77% de cópias repetidas, divididas em treino (70%), validação (15%) e teste (15%), "
    "com 17 classes: dígitos, + − × ÷ = ( ) e x. A CNN tem três blocos de convolução (32, 64 e 128 "
    "filtros) com batch normalization e foi treinada na CPU com data augmentation, dropout e early "
    "stopping. Uma versão sem essas proteções, treinada com 5% dos dados, serviu de comparação.",
    "Na webcam, o OpenCV encontra a folha e recorta cada símbolo. O agente usa o LLM "
    "qwen3:4b-instruct (Ollama) com tool calling: pergunta quando a confiança fica abaixo de 80%, "
    "calcula com o SymPy e confere se cada número da explicação está certo e veio da ferramenta "
    "(Chain of Verification).",
]

RESULTADOS = [
    "No conjunto de teste, usado uma única vez, a CNN acertou 94,5% das imagens, valor próximo ao da "
    "validação (94,9%), o que indica generalização. O modelo que decora chegou a 99,9% no treino, "
    "mas só 90,3% na validação. A maior confusão foi entre a incógnita x e o sinal ×, resolvida "
    "pelo contexto da conta. Em testes reais pela webcam, 77% dos símbolos foram lidos corretamente "
    "e, quando havia dúvida, o tutor perguntou ao usuário. Os testes também revelaram uma alucinação: "
    "após uma leitura errada, o LLM inventou uma conta que não estava na folha, com cálculos corretos. "
    "A conferência passou a rejeitar números que não vieram da ferramenta de cálculo.",
]

REFERENCIAS = [
    "DHULIAWALA, S. et al. Chain-of-Verification reduces hallucination in large language models. "
    "arXiv preprint arXiv:2309.11495, 2023.",
    "HANDWRITTEN math symbols dataset. Kaggle, [s.d.]. Disponível em: https://www.kaggle.com/"
    "datasets/xainano/handwrittenmathsymbols. Acesso em: 23 nov. 2026.",
    "LECUN, Y. et al. Gradient-based learning applied to document recognition. Proceedings of the "
    "IEEE, v. 86, n. 11, p. 2278-2324, 1998.",
    "MEURER, A. et al. SymPy: symbolic computing in Python. PeerJ Computer Science, v. 3, e103, 2017.",
]

#   = espaço que não quebra a linha: uma conta nunca fica partida entre duas linhas
EXPLICACAO_EXEMPLO = ("Primeiro fazemos a multiplicação, porque ela vem antes da soma: 7 × 3 = 21. "
                      "A conta fica 12 + 21. Agora somamos: 12 + 21 = 33. "
                      "O resultado é 33!")

# ---------------------------------------------------------------------------
# Áreas do modelo (em pixels, medidas no "Banner Workshop IA.png" de 3401 x 4535)
# ---------------------------------------------------------------------------
MODELO = RAIZ / "Banner Workshop IA.png"
AREAS = {
    "titulo":      (120, 555, 3285, 845),
    "autores":     (120, 870, 3285, 1062),
    "introducao":  (119, 1245, 1673, 2500),
    "metodologia": (119, 2722, 1673, 3700),
    "fig_metod":   (119, 3790, 1673, 4300),
    "resultados":  (1728, 1245, 3282, 2050),
    "fig_result":  (1728, 2110, 3284, 3300),
    "referencias": (1725, 3540, 3282, 4400),
}
FONTE = r"C:\Windows\Fonts\segoeui.ttf"
FONTE_NEGRITO = r"C:\Windows\Fonts\segoeuib.ttf"
FONTE_NEGRITO_ITALICO = r"C:\Windows\Fonts\segoeuiz.ttf"
COR_TEXTO = (0, 0, 0)
COR_AUTOR = (0, 74, 173)    # o azul do nome do autor no modelo
# Medidas do modelo: o nome ocupa y 897-960 e o e-mail y 997-1041; a margem direita é x = 3255
NOME_TOPO, NOME_ALTURA, EMAIL_TOPO, EMAIL_ALTURA, MARGEM_DIREITA = 897, 63, 997, 44, 3255
AZUL, LARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def figura_metodologia(caminho):
    """Diagrama do fluxo: webcam -> OpenCV -> CNN -> agente (LLM + SymPy) -> resposta."""
    fig, ax = plt.subplots(figsize=(10.4, 4.3), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 41); ax.axis("off")
    caixas = [  # (x, título, detalhe, cor)
        (0.3, "Webcam", "conta escrita\nà mão", "#f1f0ec"),
        (20.3, "OpenCV", "acha a folha,\nrecorta os\nsímbolos", "#f1f0ec"),
        (40.3, "CNN", "do zero:\n17 símbolos\n+ confiança", "#e3eefb"),
        (60.3, "Agente", "LLM local\nqwen3:4b\n(tool calling)", "#fde9e0"),
        (80.3, "Resposta", "explicação\nconferida,\ntela e voz", "#e2f4ec"),
    ]
    for x, titulo, detalhe, cor in caixas:
        ax.add_patch(FancyBboxPatch((x, 11), 18, 20, boxstyle="round,pad=0.4,rounding_size=1.8",
                                    facecolor=cor, edgecolor="#8a8984", linewidth=1.8))
        ax.text(x + 9, 27, titulo, ha="center", va="center", fontsize=17, fontweight="bold", color=TINTA)
        ax.text(x + 9, 18.2, detalhe, ha="center", va="center", fontsize=12.5, color=TINTA_2, linespacing=1.25)
    for x in (18.9, 38.9, 58.9, 78.9):  # setas entre as caixas
        ax.annotate("", xy=(x + 1.9, 21), xytext=(x, 21),
                    arrowprops=dict(arrowstyle="-|>", color=TINTA, lw=2.2, mutation_scale=24))
    # Ramos do agente: SymPy (calcula) e pergunta (confiança baixa)
    ax.add_patch(FancyBboxPatch((52.3, 0.6), 34, 6.8, boxstyle="round,pad=0.3,rounding_size=1.2",
                                facecolor="white", edgecolor=LARANJA, linewidth=1.8))
    ax.text(69.3, 4, "SymPy: cálculo exato", ha="center", va="center", fontsize=13, color=TINTA,
            fontweight="bold")
    ax.annotate("", xy=(69.3, 7.9), xytext=(69.3, 10.6),
                arrowprops=dict(arrowstyle="<|-|>", color=LARANJA, lw=2, mutation_scale=20))
    ax.add_patch(FancyBboxPatch((38.8, 34), 61, 6.4, boxstyle="round,pad=0.3,rounding_size=1.2",
                                facecolor="white", edgecolor=AZUL, linewidth=1.8))
    ax.text(69.3, 37.2, "confiança < 80% → pergunta ao visitante", ha="center", va="center",
            fontsize=13, color=TINTA, fontweight="bold")
    ax.annotate("", xy=(69.3, 33.6), xytext=(69.3, 31.6),
                arrowprops=dict(arrowstyle="-|>", color=AZUL, lw=2, mutation_scale=20))
    fig.savefig(caminho, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def _folha_exemplo(modelo):
    """Conta 12+7×3 montada com imagens da VALIDAÇÃO, lida pela segmentação + CNN."""
    from config import CLASSES, PASTA_DADOS_PROCESSADOS
    from segmentacao import classificar, desenhar_leitura, segmentar

    dados = np.load(PASTA_DADOS_PROCESSADOS / "validacao.npz")
    X, y = dados["X"], dados["y"]
    gerador = np.random.default_rng(4)
    blocos = []
    for simbolo in "12+7×3":
        # o × raramente passa de ~70% de confiança (parece demais com o x): exige menos dele
        exigencia = 0.5 if simbolo == "×" else 0.9
        for i in gerador.permutation(np.where(y == CLASSES.index(simbolo))[0]):
            bloco = cv2.resize((X[i][:, :, 0] * 255).astype(np.uint8), (135, 135), interpolation=cv2.INTER_NEAREST)
            bloco = cv2.erode(bloco, np.ones((3, 3), np.uint8))
            sozinho = classificar(segmentar(cv2.copyMakeBorder(bloco, 40, 40, 40, 40, cv2.BORDER_CONSTANT,
                                                               value=255)), modelo)
            if len(sozinho) == 1 and sozinho[0]["simbolo"] == simbolo and sozinho[0]["confianca"] > exigencia:
                blocos.append(bloco)
                break
        else:
            raise RuntimeError(f"nenhum exemplo de {simbolo} lido certo sozinho")
    linha = np.hstack([np.hstack([b, np.full((135, 50), 255, np.uint8)]) for b in blocos])
    folha = cv2.cvtColor(cv2.copyMakeBorder(linha, 30, 30, 50, 0, cv2.BORDER_CONSTANT, value=255), cv2.COLOR_GRAY2BGR)
    lidos = classificar(segmentar(folha), modelo)
    return cv2.cvtColor(desenhar_leitura(folha, [{"caixa": s["caixa"], "confianca": s["confianca"]} for s in lidos]),
                        cv2.COLOR_BGR2RGB), lidos


def figura_resultados(caminho, modelo):
    """Em cima: aprende x decora (treino -> validação -> teste). Embaixo: um exemplo de leitura."""
    fig = plt.figure(figsize=(10.2, 9.1), dpi=200)
    grade = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.22)

    ax = fig.add_subplot(grade[0])
    etapas = ["Treino", "Validação", "Teste"]
    ax.plot([0, 1, 2], [95.4, 94.9, 94.5], color=AZUL, lw=2.5, marker="o", markersize=9,
            markeredgecolor="white", markeredgewidth=2, label="modelo bom (todos os dados + proteções)")
    ax.plot([0, 1], [99.9, 90.3], color=LARANJA, lw=2.5, marker="o", markersize=9,
            markeredgecolor="white", markeredgewidth=2, label="modelo que decora (5% dos dados, sem proteções)")
    for x, v, cor, deslocamento in ((0, 95.4, AZUL, -14), (1, 94.9, AZUL, 9), (2, 94.5, AZUL, 9),
                                    (0, 99.9, LARANJA, 8), (1, 90.3, LARANJA, -16)):
        ax.annotate(f"{v:.1f}%".replace(".", ","), (x, v), xytext=(0, deslocamento), textcoords="offset points",
                    ha="center", fontsize=12, color=TINTA, fontweight="bold")
    ax.set_xticks([0, 1, 2], etapas, fontsize=12.5, color=TINTA_2)
    ax.set_xlim(-0.35, 2.35); ax.set_ylim(87.8, 101.8)
    ax.set_ylabel("acurácia (%)", fontsize=12, color=TINTA_2)
    ax.set_title("A rede aprende, não decora", loc="left", fontsize=15, fontweight="bold", color=TINTA)
    ax.grid(axis="y", color=GRADE, lw=0.9); ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2, labelsize=11)
    ax.legend(frameon=False, fontsize=11, loc="upper right")  # canto vazio do gráfico

    baixo = grade[1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.12)  # leitura em cima, resposta embaixo
    ax_img = fig.add_subplot(baixo[0])
    imagem, lidos = _folha_exemplo(modelo)
    ax_img.imshow(imagem)
    for s in lidos:  # rótulo grande acima de cada caixa: símbolo + confiança
        x, y, w, h = s["caixa"]
        cor = VERDE if s["confianca"] >= 0.8 else LARANJA
        ax_img.text(x + w / 2, y - 10, f"{s['simbolo']}  {100 * s['confianca']:.0f}%", ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color=cor)
    ax_img.set_ylim(imagem.shape[0], -60)  # espaço acima para os rótulos
    ax_img.set_title("O que a CNN leu (verde: confiante · laranja: duvidoso)", loc="left", fontsize=13,
                     fontweight="bold", color=TINTA)
    ax_img.axis("off")
    ax_txt = fig.add_subplot(baixo[1]); ax_txt.axis("off")
    ax_txt.add_patch(FancyBboxPatch((0.01, 0.03), 0.98, 0.94, boxstyle="round,pad=0.01,rounding_size=0.04",
                                    transform=ax_txt.transAxes, facecolor="#e3eefb", edgecolor=AZUL, lw=1.6))
    ax_txt.text(0.04, 0.88, "Resposta do tutor (conferida pelo SymPy ✔)", transform=ax_txt.transAxes,
                fontsize=13, fontweight="bold", color=AZUL, va="top")
    import textwrap
    ax_txt.text(0.04, 0.66, textwrap.fill(EXPLICACAO_EXEMPLO, 72), transform=ax_txt.transAxes,
                fontsize=12.5, color=TINTA, va="top", linespacing=1.45)
    fig.savefig(caminho, dpi=200, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Texto no banner (justificado, como no modelo)
# ---------------------------------------------------------------------------

def _quebrar(desenho, texto, fonte, largura, recuo=0):
    linhas, atual = [], []
    for palavra in texto.split():
        teste = " ".join(atual + [palavra])
        limite = largura - (recuo if not linhas else 0)
        if atual and desenho.textlength(teste, font=fonte) > limite:
            linhas.append(atual)
            atual = [palavra]
        else:
            atual.append(palavra)
    if atual:
        linhas.append(atual)
    return linhas


def _altura_paragrafos(desenho, paragrafos, fonte, largura, entre, recuo, espaco_par):
    total = 0
    for p in paragrafos:
        total += len(_quebrar(desenho, p, fonte, largura, recuo)) * entre + espaco_par
    return total - espaco_par


def escrever(desenho, area, paragrafos, tamanho_max=52, justificar=True, recuo_par=True, espaco_par=0.0):
    """Escreve os parágrafos na área, com o maior tamanho de letra que couber."""
    x0, y0, x1, y1 = area
    largura, altura = x1 - x0, y1 - y0
    for tamanho in range(tamanho_max, 24, -1):
        fonte = ImageFont.truetype(FONTE, tamanho)
        entre, recuo = round(tamanho * 1.5), (round(tamanho * 0.9) if recuo_par else 0)
        if _altura_paragrafos(desenho, paragrafos, fonte, largura, entre, recuo, espaco_par * entre) <= altura:
            break
    y = y0
    for p in paragrafos:
        linhas = _quebrar(desenho, p, fonte, largura, recuo)
        for n, palavras in enumerate(linhas):
            x = x0 + (recuo if n == 0 else 0)
            disponivel = largura - (recuo if n == 0 else 0)
            ultima = n == len(linhas) - 1
            if justificar and not ultima and len(palavras) > 1:
                ocupado = sum(desenho.textlength(w, font=fonte) for w in palavras)
                folga = (disponivel - ocupado) / (len(palavras) - 1)
                for w in palavras:
                    desenho.text((x, y), w, font=fonte, fill=COR_TEXTO)
                    x += desenho.textlength(w, font=fonte) + folga
            else:
                desenho.text((x, y), " ".join(palavras), font=fonte, fill=COR_TEXTO)
            y += entre
        y += espaco_par * entre
    return tamanho, round(y)  # tamanho da letra usado e onde o texto terminou


def _fonte_com_altura(caminho, exemplo, altura):
    """Tamanho de letra cujo texto de exemplo tem a altura medida no modelo."""
    for tamanho in range(30, 120):
        fonte = ImageFont.truetype(caminho, tamanho)
        _, topo, _, base = fonte.getbbox(exemplo)
        if base - topo >= altura:
            return fonte
    return fonte


def escrever_autores(desenho):
    """Autores lado a lado, alinhados à direita como no modelo: nome azul em negrito
    e, embaixo, o e-mail em negrito itálico."""
    fonte_nome = _fonte_com_altura(FONTE_NEGRITO, "Nome Completo do Autor", NOME_ALTURA)
    fonte_email = _fonte_com_altura(FONTE_NEGRITO_ITALICO, "seuemail@academico.unirv.edu.br", EMAIL_ALTURA)
    direita = MARGEM_DIREITA
    for nome, email in reversed(AUTORES):  # do último (à direita) para o primeiro
        largura = max(desenho.textlength(nome, font=fonte_nome), desenho.textlength(email, font=fonte_email))
        for texto, fonte, topo, cor in ((nome, fonte_nome, NOME_TOPO, COR_AUTOR),
                                        (email, fonte_email, EMAIL_TOPO, COR_TEXTO)):
            deslocamento = fonte.getbbox(texto)[1]  # o desenho começa um pouco acima da letra
            desenho.text((direita - desenho.textlength(texto, font=fonte), topo - deslocamento), texto,
                         font=fonte, fill=cor)
        direita -= largura + 150  # espaço entre os autores


def colar_figura(banner, area, caminho):
    x0, y0, x1, y1 = area
    figura = Image.open(caminho).convert("RGB")
    escala = min((x1 - x0) / figura.width, (y1 - y0) / figura.height)
    figura = figura.resize((round(figura.width * escala), round(figura.height * escala)), Image.LANCZOS)
    banner.paste(figura, (x0 + (x1 - x0 - figura.width) // 2, y0 + (y1 - y0 - figura.height) // 2))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import keras

    print("Gerando as figuras...")
    modelo = keras.models.load_model(RAIZ / "models" / "cnn_simbolos.keras")
    figura_metodologia(PASTA / "figura_metodologia.png")
    figura_resultados(PASTA / "figura_resultados.png", modelo)

    print("Montando o banner...")
    banner = Image.open(MODELO).convert("RGB")
    desenho = ImageDraw.Draw(banner)
    for area in AREAS.values():  # apaga os textos e figuras de exemplo do modelo
        desenho.rectangle(area, fill="white")

    # Título: 2 linhas, negrito, centralizado
    x0, y0, x1, y1 = AREAS["titulo"]
    for tamanho in range(110, 60, -2):
        fonte = ImageFont.truetype(FONTE_NEGRITO, tamanho)
        if max(desenho.textlength(t, font=fonte) for t in TITULO) <= (x1 - x0):
            break
    passo = (y1 - y0) / len(TITULO)
    for n, linha in enumerate(TITULO):
        largura = desenho.textlength(linha, font=fonte)
        desenho.text((x0 + (x1 - x0 - largura) / 2, y0 + n * passo + (passo - tamanho * 1.25) / 2), linha,
                     font=fonte, fill=COR_TEXTO)

    escrever_autores(desenho)

    tamanhos, fins = {}, {}
    for secao, paragrafos, opcoes in (
            ("introducao", INTRODUCAO, {}),
            ("metodologia", METODOLOGIA, {}),
            ("resultados", RESULTADOS, {}),
            ("referencias", REFERENCIAS, {"tamanho_max": 46, "justificar": False, "recuo_par": False,
                                          "espaco_par": 0.55})):
        tamanhos[secao], fins[secao] = escrever(desenho, AREAS[secao], paragrafos, **opcoes)

    # Cada figura vai no espaço livre logo abaixo do seu texto (sem vãos em branco)
    colar_figura(banner, (119, fins["metodologia"] + 40, 1673, 4430), PASTA / "figura_metodologia.png")
    colar_figura(banner, (1728, fins["resultados"] + 30, 3284, 3330), PASTA / "figura_resultados.png")
    banner.save(PASTA / "banner_preenchido.png", optimize=True)
    print("Tamanho da letra usado em cada seção:", tamanhos)
    print(f"Pronto: {(PASTA / 'banner_preenchido.png').relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
