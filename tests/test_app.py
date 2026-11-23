"""Testes da interface (Etapa 7), sem abrir o navegador.

Chamam as mesmas funções que os botões "Ler conta" e "Ler desenho" chamam,
com uma conta montada com imagens da VALIDAÇÃO (nunca do teste) e o LLM falso
dos testes do agente.

Rodar: python -m pytest
"""
import cv2
import numpy as np
import pytest

import app
from agente import Agente
from config import CLASSES, PASTA_DADOS_PROCESSADOS, PASTA_MODELOS
from leitor_webcam import retangulo_guia
from segmentacao import classificar, segmentar
from test_agente import LLMFalso

MODELO = PASTA_MODELOS / "cnn_simbolos.keras"
VALIDACAO = PASTA_DADOS_PROCESSADOS / "validacao.npz"
CONTA = "12+7-3"
EXPLICACAO = "Da esquerda para a direita: 12 + 7 = 19. Depois 19 - 3 = 16. O resultado é 16!"

pytestmark = pytest.mark.skipif(not (MODELO.exists() and VALIDACAO.exists()),
                                reason="modelo ou dados ainda não existem")


@pytest.fixture(scope="module")
def modelo():
    keras = pytest.importorskip("keras")
    return keras.models.load_model(MODELO)


@pytest.fixture(scope="module")
def folha(modelo):
    """Conta escrita em preto no branco, com exemplos da validação bem lidos sozinhos."""
    dados = np.load(VALIDACAO)
    X, y = dados["X"], dados["y"]
    gerador = np.random.default_rng(1)
    blocos = []
    for simbolo in CONTA:
        for i in gerador.permutation(np.where(y == CLASSES.index(simbolo))[0]):
            bloco = cv2.resize((X[i][:, :, 0] * 255).astype(np.uint8), (135, 135), interpolation=cv2.INTER_NEAREST)
            bloco = cv2.erode(bloco, np.ones((3, 3), np.uint8))       # traço de caneta
            sozinho = classificar(segmentar(cv2.copyMakeBorder(bloco, 40, 40, 40, 40, cv2.BORDER_CONSTANT,
                                                               value=255)), modelo)
            if len(sozinho) == 1 and sozinho[0]["simbolo"] == simbolo and sozinho[0]["confianca"] > 0.95:
                blocos.append(bloco)
                break
    espaco = np.full((135, 45), 255, np.uint8)
    linha = np.hstack([np.hstack([b, espaco]) for b in blocos])
    return cv2.copyMakeBorder(linha, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)


def agente_falso():
    return Agente(chat=LLMFalso(("calcular", {"expressao": CONTA}), EXPLICACAO))


def test_aba_desenhar_le_e_explica(folha, modelo):
    # O quadro de desenho do Gradio entrega RGBA: fundo transparente, traço preto
    rgba = np.zeros((*folha.shape, 4), np.uint8)
    rgba[..., 3] = 255 - folha
    passos = list(app.ler_desenho({"composite": rgba}, agente_falso(), [], modelo_cnn=modelo))

    anotada, leitura, chat, _ = passos[0]           # 1º: a leitura aparece na hora
    assert chat[-1]["content"] == app.PENSANDO
    assert chat[-2]["content"] == f"Minha conta: **{CONTA}**"
    assert anotada.shape == (*folha.shape, 3)
    assert leitura.count("class='chip'") + leitura.count("class='chip duvida'") == len(CONTA)

    _, _, chat, painel = passos[-1]                 # 2º: a explicação do agente
    assert chat[-1]["content"] == EXPLICACAO
    assert "`calcular`" in painel and "✅ aprovada" in painel


def test_aba_webcam_le_so_a_folha(folha, modelo, monkeypatch):
    monkeypatch.setattr(app, "salvar_captura", lambda *a: None)   # não grava em capturas/
    quadro = np.full((720, 1280, 3), (130, 110, 90), np.uint8)   # parede atrás (RGB)
    x, y, w, h = retangulo_guia(quadro)
    folha = cv2.resize(folha, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    alto, largo = folha.shape
    topo, esquerda = y + (h - alto) // 2, x + (w - largo) // 2
    quadro[topo:topo + alto, esquerda:esquerda + largo] = cv2.cvtColor(np.minimum(folha, 225), cv2.COLOR_GRAY2RGB)
    quadro[0:150, 0:220] = 20                                    # objeto escuro no canto
    cv2.rectangle(quadro, (esquerda + largo - 40, topo + 40), (1280, topo + 110), (220, 160, 120), -1)  # "dedos"
    rgb = quadro

    passos = list(app.ler_webcam(rgb, agente_falso(), [], modelo_cnn=modelo))
    assert passos[0][2][-2]["content"] == f"Minha conta: **{CONTA}**"   # a "mão" foi ignorada
    assert passos[-1][2][-1]["content"] == EXPLICACAO


def test_sem_imagem_avisa_sem_chamar_o_agente(modelo):
    llm = LLMFalso()
    vazio = np.zeros((320, 1000, 4), np.uint8)
    chat = list(app.ler_desenho({"composite": vazio}, Agente(chat=llm), [], modelo_cnn=modelo))[-1][2]
    assert "vazio" in chat[-1]["content"]
    chat = list(app.ler_webcam(None, Agente(chat=llm), [], modelo_cnn=modelo))[-1][2]
    assert "ligue a câmera" in chat[-1]["content"]
    assert llm.recebidas == []


def test_conversa_pelo_chat():
    agente = Agente(chat=LLMFalso("Porque a multiplicação vem antes da soma."))
    passos = list(app.conversar("por que?", agente, []))
    assert passos[0][1][-1]["content"] == app.PENSANDO
    assert passos[-1][1][-1]["content"] == "Porque a multiplicação vem antes da soma."


def test_simbolo_duvidoso_fica_laranja():
    html = app.html_leitura([{"simbolo": "7", "confianca": 0.55, "top3": [("7", 0.55), ("1", 0.4)]},
                             {"simbolo": "+", "confianca": 0.99, "top3": [("+", 0.99)]}])
    assert html.count("chip duvida") == 1 and "55%" in html


def test_interface_monta_sem_erro(modelo):
    assert app.criar_interface(modelo) is not None
