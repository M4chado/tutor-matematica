"""Testes da segmentação (encontrar e separar os símbolos de uma conta).

Rodar: python -m pytest
"""
import cv2
import numpy as np
import pytest

from config import CLASSES, PASTA_DADOS_PROCESSADOS, PASTA_MODELOS
from segmentacao import classificar, mascara_da_folha, montar_expressao, segmentar

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


# ---------------------------------------------------------------------------
# Detecção da folha: só vale a tinta cercada de papel
# (cena sintética imitando uma foto com a folha segurada na mão)
# ---------------------------------------------------------------------------

def cena_com_folha(folha_cinza):
    """Folha branca com a conta no meio de uma cena bagunçada (BGR)."""
    gerador = np.random.default_rng(5)
    cena = np.full((720, 1280, 3), (90, 110, 130), np.uint8)                 # parede/fundo
    for _ in range(25):                                                       # "prateleira" colorida
        x, y = int(gerador.integers(950, 1250)), int(gerador.integers(0, 690))
        cor = tuple(int(c) for c in gerador.integers(0, 255, 3))
        cv2.rectangle(cena, (x, y), (x + 30, y + 25), cor, -1)
    cv2.rectangle(cena, (0, 0), (230, 720), (235, 235, 235), -1)              # "camisa branca"
    cv2.putText(cena, "LOGO", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (40, 40, 220), 5)  # logotipo vermelho
    cv2.line(cena, (60, 500), (120, 650), (60, 60, 60), 6)                    # dobra escura na camisa
    alto, largo = folha_cinza.shape
    y0, x0 = (720 - alto) // 2, 250
    cena[y0:y0 + alto, x0:x0 + largo] = cv2.cvtColor(folha_cinza, cv2.COLOR_GRAY2BGR)
    cv2.ellipse(cena, (x0 + largo, y0 + alto // 2), (60, 35), 0, 0, 360, (120, 160, 220), -1)  # "dedo"
    return cena


@pytest.fixture(scope="module")
def conta_validacao(modelo):
    """Folha (cinza, traço preto) com 12+7-3 escrito com exemplos da validação."""
    dados = np.load(VALIDACAO)
    X, y = dados["X"], dados["y"]
    gerador = np.random.default_rng(2)
    blocos = []
    for simbolo in "12+7-3":
        for i in gerador.permutation(np.where(y == CLASSES.index(simbolo))[0]):
            bloco = cv2.resize(como_tinta_de_caneta(X[i]), (100, 100), interpolation=cv2.INTER_AREA)
            sozinho = classificar(segmentar(cv2.copyMakeBorder(bloco, 40, 40, 40, 40, cv2.BORDER_CONSTANT,
                                                               value=255)), modelo)
            if len(sozinho) == 1 and sozinho[0]["simbolo"] == simbolo and sozinho[0]["confianca"] > 0.95:
                blocos.append(bloco)
                break
    linha = np.hstack([np.hstack([b, np.full((100, 30), 255, np.uint8)]) for b in blocos])
    return np.minimum(cv2.copyMakeBorder(linha, 80, 80, 60, 60, cv2.BORDER_CONSTANT, value=255), 245)


@pytest.mark.skipif(not (MODELO.exists() and VALIDACAO.exists()), reason="modelo ou dados ainda não existem")
def test_folha_ignora_tudo_em_volta(conta_validacao, modelo):
    cena = cena_com_folha(conta_validacao)
    sem_folha = segmentar(cena)
    com_folha = classificar(segmentar(cena, so_na_folha=True), modelo)
    assert montar_expressao(com_folha) == "12+7-3"
    assert len(sem_folha) > len(com_folha)        # sem a detecção, o fundo vira "símbolo"


def test_folha_nao_encontrada_devolve_none():
    escura = np.full((480, 640, 3), 40, np.uint8)
    assert mascara_da_folha(escura, escura.shape[:2]) is None


def test_caneta_azul_vale_e_tinta_vermelha_nao():
    folha = np.full((400, 600, 3), 240, np.uint8)
    cv2.line(folha, (100, 200), (200, 200), (160, 40, 30), 8)   # azul (BGR)
    cv2.line(folha, (400, 150), (400, 250), (40, 40, 210), 8)   # vermelho (BGR)
    cena = cv2.copyMakeBorder(folha, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(60, 70, 80))
    simbolos = segmentar(cena, so_na_folha=True)
    assert len(simbolos) == 1
    x, _, w, _ = simbolos[0]["caixa"]
    assert 180 < x + w / 2 < 330                                  # é o traço azul


@pytest.mark.skipif(not (MODELO.exists() and VALIDACAO.exists()), reason="modelo ou dados ainda não existem")
def test_folha_com_luz_fraca_de_um_lado_cabelo_e_marcas_do_verso(conta_validacao, modelo):
    """Reproduz o ensaio sem Wi-Fi que falhou: luz fraca e desigual, a cabeça de quem
    segura a folha aparecendo atrás dela e marcas fracas do outro lado do papel."""
    cena = np.full((720, 1280, 3), (150, 165, 175), np.uint8)                # parede bege
    cv2.ellipse(cena, (640, 150), (110, 70), 0, 0, 360, (25, 25, 30), -1)   # cabeça (cabelo escuro)
    alto, largo = conta_validacao.shape
    y0, x0 = 170, (1280 - largo) // 2
    folha = cv2.cvtColor(conta_validacao, cv2.COLOR_GRAY2BGR)
    cv2.line(folha, (30, 40), (120, 40), (190, 190, 190), 4)                   # marca fraca do verso
    cv2.circle(folha, (largo - 60, alto - 40), 18, (185, 185, 185), 3)        # outra marca fraca
    cena[y0:y0 + alto, x0:x0 + largo] = folha
    cv2.ellipse(cena, (640, 150 + 40), (90, 40), 0, 180, 360, (25, 25, 30), -1)  # cabelo por cima da borda
    # Luz fraca vindo da direita: a esquerda fica com 45% do brilho
    luz = np.linspace(0.45, 1.0, 1280, dtype=np.float32)[None, :, None]
    cena = (cena.astype(np.float32) * luz).astype(np.uint8)

    lidos = classificar(segmentar(cena, so_na_folha=True), modelo)
    assert montar_expressao(lidos) == "12+7-3"


@pytest.mark.skipif(not (MODELO.exists() and VALIDACAO.exists()), reason="modelo ou dados ainda não existem")
def test_marca_solta_longe_da_conta_e_ignorada(conta_validacao, modelo):
    """Reproduz o erro da pasta "provas de erros": uma marca escura na altura da
    conta, mas longe dela (um objeto da prateleira ao lado da folha), virava um "+"."""
    alto = conta_validacao.shape[0]
    folha = cv2.copyMakeBorder(conta_validacao, 0, 0, 500, 0, cv2.BORDER_CONSTANT, value=245)
    meio = alto // 2
    cv2.line(folha, (80, meio - 15), (80, meio + 15), 30, 6)     # marquinha solta, bem à esquerda
    cv2.line(folha, (65, meio), (95, meio), 30, 6)
    cena = cv2.copyMakeBorder(cv2.cvtColor(folha, cv2.COLOR_GRAY2BGR), 150, 150, 80, 80,
                              cv2.BORDER_CONSTANT, value=(90, 110, 130))
    assert montar_expressao(classificar(segmentar(cena, so_na_folha=True), modelo)) == "12+7-3"


def test_preto_amarronzado_com_a_folha_perto_da_camera_vale():
    """Reproduz o "9 - 3 - 2" lido como "932": com a folha perto da câmera, o preto da
    caneta puxou para o marrom (saturação 73) e os sinais de menos eram descartados
    como se fossem pele. Tinta quase preta é caneta, qualquer que seja a cor."""
    folha = np.full((400, 600, 3), 235, np.uint8)
    cv2.line(folha, (250, 200), (350, 200), (25, 35, 45), 10)    # "−" preto amarronzado (BGR)
    cena = cv2.copyMakeBorder(folha, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(60, 70, 80))
    assert len(segmentar(cena, so_na_folha=True)) == 1
