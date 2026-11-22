"""Testes da segmentação (encontrar e separar os símbolos de uma conta).

Rodar: python -m pytest
"""
import cv2
import numpy as np
import pytest

from config import CLASSES, PASTA_DADOS_PROCESSADOS, PASTA_MODELOS
from segmentacao import classificar, montar_expressao, segmentar

TINTA, ESPESSURA = 30, 9


def papel(altura=300, largura=900, sombra=False):
    img = np.full((altura, largura), 225, np.float32)
    if sombra:  # escurece a esquerda até 55%
        img *= np.linspace(0.55, 1.0, largura, dtype=np.float32)[None, :]
    return img


def um(img, x):
    cv2.line(img, (x + 20, 80), (x + 25, 220), TINTA, ESPESSURA)


def mais(img, x):
    cv2.line(img, (x + 50, 90), (x + 50, 210), TINTA, ESPESSURA)
    cv2.line(img, (x, 150), (x + 100, 150), TINTA, ESPESSURA)


def sete(img, x):
    cv2.line(img, (x, 80), (x + 90, 80), TINTA, ESPESSURA)
    cv2.line(img, (x + 90, 80), (x + 30, 220), TINTA, ESPESSURA)


def igual(img, x):
    cv2.line(img, (x, 125), (x + 100, 125), TINTA, ESPESSURA)
    cv2.line(img, (x, 175), (x + 100, 175), TINTA, ESPESSURA)


def dividir(img, x):
    cv2.line(img, (x, 150), (x + 100, 150), TINTA, ESPESSURA)
    cv2.circle(img, (x + 50, 105), 9, TINTA, -1)
    cv2.circle(img, (x + 50, 195), 9, TINTA, -1)


def cinco_com_chapeu_solto(img, x):
    cv2.line(img, (x + 10, 80), (x + 90, 80), TINTA, ESPESSURA)           # chapéu, separado
    cv2.line(img, (x + 10, 100), (x + 10, 145), TINTA, ESPESSURA)
    cv2.ellipse(img, (x + 45, 180), (45, 40), 0, -120, 150, TINTA, ESPESSURA)


def finalizar(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def test_conta_simples_quantidade_e_ordem():
    img = papel()
    um(img, 50); mais(img, 200); sete(img, 400)
    simbolos = segmentar(finalizar(img))
    assert len(simbolos) == 3
    xs = [s["caixa"][0] for s in simbolos]
    assert xs == sorted(xs)                     # da esquerda para a direita


def test_igual_vira_um_simbolo():
    img = papel(); igual(img, 300)
    assert len(segmentar(finalizar(img))) == 1


def test_dividir_com_pontos_vira_um_simbolo():
    img = papel(); dividir(img, 300)
    simbolos = segmentar(finalizar(img))
    assert len(simbolos) == 1
    _, _, _, altura = simbolos[0]["caixa"]
    assert altura > 80                          # a caixa inclui os dois pontos


def test_cinco_com_chapeu_solto_vira_um_simbolo():
    img = papel(); cinco_com_chapeu_solto(img, 300)
    assert len(segmentar(finalizar(img))) == 1


def test_sujeira_e_ignorada():
    img = papel()
    um(img, 50); mais(img, 200); sete(img, 400)
    for x, y in [(700, 60), (750, 250), (320, 40), (600, 270)]:   # pontinhos de sujeira
        cv2.circle(img, (x, y), 2, TINTA, -1)
    assert len(segmentar(finalizar(img))) == 3


def test_borda_escura_e_ignorada():
    """Uma faixa escura encostada na borda (mesa, dedo) não é símbolo."""
    img = papel()
    img[:, :40] = 40
    mais(img, 200)
    assert len(segmentar(finalizar(img))) == 1


def test_sombra_nao_atrapalha():
    img = papel(sombra=True)
    um(img, 50); igual(img, 200); dividir(img, 400); sete(img, 600)
    assert len(segmentar(finalizar(img))) == 4


def test_papel_em_branco_nao_tem_simbolos():
    assert segmentar(finalizar(papel())) == []


def test_entrada_pronta_para_a_cnn():
    img = papel(); um(img, 50); mais(img, 200)
    for s in segmentar(finalizar(img)):
        assert s["entrada"].shape == (45, 45, 1)
        assert s["entrada"].dtype == np.float32


def test_caixas_na_escala_da_imagem_original():
    """Fotos grandes são reduzidas por dentro, mas as caixas voltam ao tamanho original."""
    img = cv2.resize(papel(), (2700, 900))
    mais(img, 1500)  # desenhado já na escala grande (traço mais fino em proporção, tudo bem)
    simbolos = segmentar(finalizar(img))
    assert len(simbolos) == 1
    x, _, w, _ = simbolos[0]["caixa"]
    assert abs(x - 1500) < 15 and abs(w - 100) < 20


# ---------------------------------------------------------------------------
# Integração: segmentação + CNN, montando uma "conta" com imagens da VALIDAÇÃO
# (nunca do teste, que é intocável)
# ---------------------------------------------------------------------------

MODELO = PASTA_MODELOS / "cnn_simbolos.keras"
VALIDACAO = PASTA_DADOS_PROCESSADOS / "validacao.npz"


@pytest.fixture(scope="module")
def modelo():
    keras = pytest.importorskip("keras")
    return keras.models.load_model(MODELO)


def como_tinta_de_caneta(imagem45):
    """Aumenta 3x e engrossa o traço, imitando uma caneta no papel."""
    grande = cv2.resize((imagem45[:, :, 0] * 255).astype(np.uint8), (135, 135), interpolation=cv2.INTER_NEAREST)
    return cv2.erode(grande, np.ones((3, 3), np.uint8))  # erode no branco = engrossa o preto


@pytest.mark.skipif(not (MODELO.exists() and VALIDACAO.exists()), reason="modelo ou dados ainda não existem")
def test_conta_montada_com_imagens_da_validacao_e_lida_certo(modelo):
    dados = np.load(VALIDACAO)
    X, y = dados["X"], dados["y"]
    gerador = np.random.default_rng(0)

    # Para cada símbolo da conta, escolhe um exemplo que, SOZINHO, já é lido certo
    # (assim o teste mede a segmentação, não a acurácia da CNN)
    conta = ["1", "2", "+", "7", "×", "3", "-", "(", "8", "÷", "4", ")", "=", "9"]
    blocos = []
    for simbolo in conta:
        rotulo = CLASSES.index(simbolo)
        for i in gerador.permutation(np.where(y == rotulo)[0]):
            bloco = como_tinta_de_caneta(X[i])
            sozinho = classificar(segmentar(cv2.copyMakeBorder(bloco, 40, 40, 40, 40,
                                                               cv2.BORDER_CONSTANT, value=255)), modelo)
            # > 0.5: o × raramente passa de ~70% de confiança (é parecido demais com o x)
            if len(sozinho) == 1 and sozinho[0]["simbolo"] == simbolo and sozinho[0]["confianca"] > 0.5:
                blocos.append(bloco)
                break
        else:
            pytest.skip(f"nenhum exemplo de {simbolo} lido certo sozinho")

    espaco = np.full((135, 45), 255, np.uint8)
    linha = np.hstack([np.hstack([b, espaco]) for b in blocos])
    folha = cv2.copyMakeBorder(linha, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)

    lidos = classificar(segmentar(folha), modelo)
    assert montar_expressao(lidos) == "".join(conta)
