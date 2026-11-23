"""Etapa 7: a interface do Tutor (Gradio, roda só no computador: http://localhost:7860).

Como rodar (com o .venv ativado e o Ollama aberto):
    python src/app.py

Abas:
  - Webcam:   liga a câmera, mostra a folha com a conta e clica em "Ler conta".
              O programa acha a folha sozinho e ignora mão, roupa e fundo.
  - Desenhar: plano B, desenha a conta com o mouse e clica em "Ler desenho".

As duas abas usam o MESMO caminho: segmentação -> CNN -> agente.
A CNN e o LLM são carregados uma vez só, quando o programa abre.
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # nada de enviar dados para a internet

import argparse  # noqa: E402
import html  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from functools import partial  # noqa: E402

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from agente import Agente  # noqa: E402
from config import LIMITE_CONFIANCA, MODELO_LLM  # noqa: E402
from ferramentas import CRUZES  # noqa: E402
from leitor_webcam import carregar_modelo, ler_quadro, salvar_captura  # noqa: E402
from segmentacao import desenhar_leitura, ler_conta, montar_expressao  # noqa: E402

TITULO = "Tutor de Matemática que Lê sua Letra"
PENSANDO = "🤔 Pensando e conferindo a conta..."

# Fontes do próprio Windows (o padrão do Gradio baixa fontes da internet; na feira não tem)
TEMA = gr.themes.Soft(primary_hue="blue", neutral_hue="slate", text_size="lg",
                      font=["Segoe UI", "Arial", "sans-serif"], font_mono=["Consolas", "monospace"])

CSS = """
.gradio-container { max-width: 1400px !important; margin: auto; }
#titulo h1 { font-size: 2.3em; margin-bottom: 0; }
#titulo p { font-size: 1.15em; opacity: .8; margin-top: .3em; }
#chat .message { font-size: 1.25em !important; line-height: 1.5; }
#chat .message strong { font-size: 1.05em; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-end; }
.chip { display: flex; flex-direction: column; align-items: center; min-width: 54px; padding: 4px 6px 6px;
        border-radius: 10px; border: 3px solid #1baf7a; }
.chip.duvida { border-color: #eb6834; border-style: dashed; }
.chip .sim { font-size: 2.4em; font-weight: 700; line-height: 1.1; }
.chip .conf { font-size: .95em; opacity: .85; }
.chip.duvida .conf { color: #eb6834; font-weight: 700; opacity: 1; }
.legenda { font-size: .95em; opacity: .8; margin-top: 6px; }
.expressao { font-size: 1.6em; font-weight: 700; margin: 4px 0 8px; }
/* Retângulo-guia por cima do vídeo da webcam: onde mostrar a folha. Se o programa
   não achar a folha, lê só esta área (80% x 40% no centro, igual a retangulo_guia) */
#camera .wrap:has(> video[data-testid="webcam-video"]) { position: relative; }
#camera video[data-testid="webcam-video"] { width: 100%; height: auto; display: block; }
#camera .wrap:has(> video[data-testid="webcam-video"])::after {
    content: "mostre a folha aqui"; position: absolute; left: 10%; top: 30%; width: 80%; height: 40%;
    box-sizing: border-box; border: 4px dashed #ffa028; border-radius: 8px; color: #ffa028; font-weight: 700;
    pointer-events: none; display: flex; align-items: flex-start; justify-content: center; }
"""


# ---------------------------------------------------------------------------
# Leitura da imagem (o mesmo caminho para a webcam e para o desenho)
# ---------------------------------------------------------------------------

def html_leitura(simbolos):
    """Cada símbolo num "cartão" com a confiança: verde = confiante, laranja = duvidoso."""
    if not simbolos:
        return "<p class='legenda'>Nenhum símbolo encontrado ainda.</p>"
    cartoes = []
    for s in simbolos:
        duvida = s["confianca"] < LIMITE_CONFIANCA
        cartoes.append(f"<div class='chip{' duvida' if duvida else ''}' title='outras: "
                       + html.escape(", ".join(f"{a} {100 * p:.0f}%" for a, p in s["top3"][1:]))
                       + f"'><span class='sim'>{html.escape(s['simbolo'])}</span>"
                       f"<span class='conf'>{100 * s['confianca']:.0f}%</span></div>")
    return (f"<div class='expressao'>{html.escape(montar_expressao(simbolos))}</div>"
            f"<div class='chips'>{''.join(cartoes)}</div>"
            f"<div class='legenda'>Confiança da rede neural em cada símbolo. "
            f"Verde: confiante. Laranja: duvidoso (abaixo de {100 * LIMITE_CONFIANCA:.0f}%): o tutor pergunta, "
            f"ou decide pelo contexto se a dúvida for só entre x e ×.</div>")


def painel_agente(resposta, simbolos=None):
    """Texto do painel "Por dentro do agente": o que ele percebeu, decidiu e conferiu."""
    linhas = []
    if simbolos is not None:
        linhas.append("**1. Percebeu (CNN):** " + " · ".join(f"`{s['simbolo']}` {100 * s['confianca']:.0f}%"
                                                              for s in simbolos))
    decisao = {"pergunta": "perguntar ao visitante (há símbolo duvidoso ou dúvida na conversa)",
               "erro": "avisar que a conta está mal escrita (não dá para calcular)",
               "exercicio": "criar um exercício", "explicacao": "calcular e explicar"}[resposta["tipo"]]
    linhas.append(f"**2. Decidiu:** {decisao}")
    if simbolos and resposta["tipo"] != "pergunta":
        pelo_contexto = [f"`{s['simbolo']}` ({100 * s['confianca']:.0f}%)" for s in simbolos
                         if s["confianca"] < LIMITE_CONFIANCA and s["simbolo"] in CRUZES]
        if pelo_contexto:
            linhas.append("   - duvidoso(s) " + ", ".join(pelo_contexto) + ", mas a dúvida era só entre x e ×: "
                          "o contexto decidiu, sem precisar perguntar")
    if resposta.get("expressao"):
        linhas.append(f"**Conta usada:** `{resposta['expressao']}`")
    if resposta.get("ferramentas"):
        linhas.append("**3. Ferramentas usadas:** " + ", ".join(f"`{f}`" for f in resposta["ferramentas"]))
    v = resposta.get("verificacao")
    if v:
        estado = "✅ aprovada" if v["aprovada"] else "⚠️ reprovada (usei a explicação automática)"
        linhas.append(f"**4. Conferência com o SymPy:** {estado}"
                      + (f" na {v['tentativas']}ª tentativa" if v["tentativas"] else ""))
        for rascunho in v["reprovadas"]:
            linhas.append(f"   - explicação descartada: _{rascunho[:120]}..._")
        for problema in v["problemas"]:
            linhas.append(f"   - motivo: {problema}")
        if v["aviso"]:
            linhas.append(f"   - aviso: {v['aviso']}")
    if resposta.get("veredito"):
        linhas.append(f"**Resposta do visitante conferida pelo código:** {resposta['veredito']['resposta_visitante']} → "
                      + ("acertou ✅" if resposta["veredito"]["acertou"] else "errou ❌"))
    calculo = resposta.get("calculo")
    if calculo and calculo.get("ok"):
        linhas.append("**Passos calculados pelo SymPy:**")
        linhas += [f"   {i}. `{p['conta']}` ({p['regra']})" for i, p in enumerate(calculo["passos"])]
    tempo = f"**Tempo:** {resposta['tempo']} s"
    if resposta.get("chamadas_llm"):
        tempo += f" · **chamadas ao LLM:** {resposta['chamadas_llm']}"
    linhas.append(tempo)
    return "\n\n".join(linhas)


def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# O que cada botão faz. São "geradores": mostram a leitura da CNN na hora
# e a explicação quando o agente termina (o LLM leva uns 15 s).
# ---------------------------------------------------------------------------

def explicar_leitura(simbolos, anotada_bgr, agente, chat):
    """Parte comum das duas abas: mostra a leitura da CNN e chama o agente."""
    anotada = _rgb(anotada_bgr)
    leitura = html_leitura(simbolos)

    if not simbolos:
        chat = chat + [{"role": "assistant", "content":
                        "Não encontrei nenhum símbolo. Escreva com traço escuro e grosso e tente de novo."}]
        yield anotada, leitura, chat, gr.skip()
        return
    chat = chat + [{"role": "user", "content": f"Minha conta: **{montar_expressao(simbolos)}**"},
                   {"role": "assistant", "content": PENSANDO}]
    yield anotada, leitura, chat, gr.skip()
    resposta = agente.ler_conta(simbolos)
    chat = chat[:-1] + [{"role": "assistant", "content": resposta["texto"]}]  # troca o "Pensando..."
    yield anotada, leitura, chat, painel_agente(resposta, simbolos)


def ler_webcam(quadro, agente, chat, modelo_cnn=None):
    """Botão "Ler conta": usa o último quadro da webcam. Lê a tinta que está na
    folha (mão e fundo são ignorados); sem folha à vista, lê o retângulo-guia."""
    if quadro is None:
        chat = chat + [{"role": "assistant", "content":
                        "Primeiro ligue a câmera: clique no quadro da câmera e permita o acesso."}]
        yield gr.skip(), gr.skip(), chat, gr.skip()
        return
    bgr = cv2.cvtColor(np.ascontiguousarray(quadro[..., :3]), cv2.COLOR_RGB2BGR)
    simbolos, anotada, _ = ler_quadro(bgr, modelo_cnn)
    salvar_captura(bgr, anotada)  # guarda em capturas/ (útil para os testes com pessoas)
    yield from explicar_leitura(simbolos, anotada, agente, chat)


def desenho_para_bgr(desenho):
    """Valor do quadro de desenho -> imagem BGR com fundo branco (None se vazio)."""
    imagem = desenho.get("composite") if isinstance(desenho, dict) else desenho
    if imagem is None:
        return None
    if imagem.ndim == 3 and imagem.shape[-1] == 4:  # fundo transparente -> papel branco
        if imagem[..., 3].max() == 0:
            return None
        alfa = imagem[..., 3:4].astype(np.float32) / 255
        imagem = (imagem[..., :3] * alfa + 255 * (1 - alfa)).astype(np.uint8)
    elif imagem.ndim == 2:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(np.ascontiguousarray(imagem), cv2.COLOR_RGB2BGR)


def ler_desenho(desenho, agente, chat, modelo_cnn=None):
    """Botão "Ler desenho": mesmo caminho da webcam, com a imagem inteira."""
    bgr = desenho_para_bgr(desenho)
    if bgr is None:
        chat = chat + [{"role": "assistant", "content": "O quadro está vazio. Desenhe a conta com o mouse."}]
        yield gr.skip(), gr.skip(), chat, gr.skip()
        return
    simbolos = ler_conta(bgr, modelo_cnn)  # o quadro de desenho já é uma "folha" limpa
    yield from explicar_leitura(simbolos, desenhar_leitura(bgr, simbolos), agente, chat)


def conversar(mensagem, agente, chat):
    """Caixa de texto e atalhos: o visitante conversa com o agente."""
    mensagem = (mensagem or "").strip()
    if not mensagem:
        yield "", chat, gr.skip()
        return
    chat = chat + [{"role": "user", "content": mensagem}, {"role": "assistant", "content": PENSANDO}]
    yield "", chat, gr.skip()
    resposta = agente.responder(mensagem)
    chat = chat[:-1] + [{"role": "assistant", "content": resposta["texto"]}]  # troca o "Pensando..."
    yield "", chat, painel_agente(resposta)


def recomecar(agente):
    """Botão "Nova conta": limpa a conversa e a tela."""
    agente.nova_conversa()
    return [], None, html_leitura([]), "Ainda não houve nenhuma leitura.", None


# ---------------------------------------------------------------------------
# A tela
# ---------------------------------------------------------------------------

def criar_interface(modelo_cnn):
    """Monta a tela. modelo_cnn: a CNN já carregada (carregada uma vez só)."""
    with gr.Blocks(title=TITULO, analytics_enabled=False) as interface:
        agente = gr.State(Agente)  # um agente (com o seu histórico) para cada aba do navegador
        quadro_webcam = gr.State(None)

        gr.Markdown(f"# 🧮 {TITULO}\nEscreva uma conta no papel, mostre para a câmera e o tutor **lê**, "
                    f"**resolve** e **explica** passo a passo.", elem_id="titulo")

        with gr.Row(equal_height=False):
            # ----- Coluna da esquerda: entrada (webcam ou desenho) -----
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("📷 Webcam"):
                        camera = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Câmera",
                                          elem_id="camera",
                                          webcam_options=gr.WebcamOptions(
                                              mirror=False,  # espelhar inverteria a escrita!
                                              constraints={"video": {"width": {"ideal": 1280},
                                                                     "height": {"ideal": 720}}}))
                        botao_webcam = gr.Button("📷 Ler conta", variant="primary", size="lg")
                    with gr.Tab("✏️ Desenhar"):
                        quadro = gr.Sketchpad(type="numpy", label="Desenhe a conta com o mouse",
                                              canvas_size=(1000, 320), height=380, layers=False,
                                              brush=gr.Brush(colors=["#111111"], default_size=7,
                                                             color_mode="fixed"))
                        botao_desenho = gr.Button("✏️ Ler desenho", variant="primary", size="lg")
                leitura_imagem = gr.Image(label="O que a rede neural leu", interactive=False, height=300,
                                          type="numpy")
                leitura_html = gr.HTML(html_leitura([]))

            # ----- Coluna da direita: conversa com o tutor -----
            with gr.Column(scale=6):
                chat = gr.Chatbot(label="Tutor", elem_id="chat", height=520, latex_delimiters=[],
                                  placeholder="Mostre uma conta para começar! 👋")
                with gr.Row():
                    texto = gr.Textbox(placeholder="Pergunte ao tutor... (ex.: por que fez a multiplicação primeiro?)",
                                       show_label=False, scale=5, autofocus=False)
                    enviar = gr.Button("Enviar", scale=1)
                with gr.Row():
                    atalhos = [gr.Button(t, size="sm") for t in
                               ("Por que nessa ordem?", "Me dá um exercício parecido", "Explica de novo, mais devagar")]
                    nova = gr.Button("🔄 Nova conta", size="sm", variant="stop")
                with gr.Accordion("🔍 Por dentro do agente (para o professor)", open=False):
                    painel = gr.Markdown("Ainda não houve nenhuma leitura.")

        # ----- Eventos -----
        camera.stream(lambda q: q, inputs=camera, outputs=quadro_webcam, stream_every=0.5,
                      show_progress="hidden", api_visibility="private")
        saidas_leitura = [leitura_imagem, leitura_html, chat, painel]
        # partial: entrega a CNN (carregada uma vez só) às funções dos botões
        botao_webcam.click(partial(ler_webcam, modelo_cnn=modelo_cnn), [quadro_webcam, agente, chat], saidas_leitura,
                           api_name="ler_webcam")
        botao_desenho.click(partial(ler_desenho, modelo_cnn=modelo_cnn), [quadro, agente, chat], saidas_leitura,
                            api_name="ler_desenho")
        texto.submit(conversar, [texto, agente, chat], [texto, chat, painel], api_name="conversar")
        enviar.click(conversar, [texto, agente, chat], [texto, chat, painel], api_visibility="private")
        for atalho in atalhos:
            atalho.click(conversar, [atalho, agente, chat], [texto, chat, painel], api_visibility="private")
        nova.click(recomecar, agente, [chat, leitura_imagem, leitura_html, painel, quadro], api_name="nova_conta")

    return interface


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Abre a interface do tutor no navegador.")
    parser.add_argument("--sem-navegador", action="store_true", help="não abre o navegador sozinho")
    args = parser.parse_args()
    inicio = time.perf_counter()
    modelo_cnn = carregar_modelo()                     # 1x só, na abertura
    print(f"Carregando o {MODELO_LLM} no Ollama...", end=" ", flush=True)
    print("pronto" if Agente().aquecer() else "FALHOU (abra o Ollama; o tutor usará a explicação automática)")
    print(f"Tudo carregado em {time.perf_counter() - inicio:.0f} s. Abrindo http://localhost:7860 ...")
    criar_interface(modelo_cnn).launch(server_name="127.0.0.1", server_port=7860, inbrowser=not args.sem_navegador,
                                       theme=TEMA, css=CSS)


if __name__ == "__main__":
    main()
