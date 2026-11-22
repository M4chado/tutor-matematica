"""Testes da função única de pré-processamento.

Rodar: python -m pytest
"""
import cv2
import numpy as np
import pytest

from config import CLASSES, MAPA_PASTAS, PASTA_IMAGENS_BRUTAS
from preprocessamento import carregar_imagem, preprocessar


def desenhar_sete(altura, largura, espessura, fundo=255, tinta=0):
    """Cria uma imagem sintética com um '7' desenhado."""
    img = np.full((altura, largura), fundo, np.uint8)
    cv2.line(img, (largura // 6, altura // 6), (largura * 5 // 6, altura // 6), tinta, espessura)
    cv2.line(img, (largura * 5 // 6, altura // 6), (largura // 3, altura * 9 // 10), tinta, espessura)
    return img


def test_formato_tipo_e_valores():
    """A saída precisa ter o formato que a CNN espera: (45, 45, 1), float32, de 0 a 1."""
    saida = preprocessar(desenhar_sete(45, 45, 1))
    assert saida.shape == (45, 45, 1)
    assert saida.dtype == np.float32
    assert saida.min() >= 0.0 and saida.max() <= 1.0


def test_polaridade_fundo_claro_traco_escuro():
    """Igual ao dataset: a maior parte é fundo (1.0) e o traço é escuro (0.0)."""
    saida = preprocessar(desenhar_sete(45, 45, 1))
    assert (saida == 1.0).mean() > 0.8   # fundo ocupa a maior parte
    assert (saida == 0.0).sum() > 20     # existe traço


def test_imagem_invertida_da_o_mesmo_resultado():
    """Traço claro em fundo escuro deve virar exatamente a mesma entrada."""
    normal = desenhar_sete(200, 160, 12)
    invertida = 255 - normal
    assert np.array_equal(preprocessar(normal), preprocessar(invertida))


def test_imagem_colorida_funciona():
    """Fotos da webcam chegam coloridas (BGR); o resultado deve ser igual ao da versão cinza."""
    cinza = desenhar_sete(200, 160, 12)
    colorida = cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)
    assert np.array_equal(preprocessar(cinza), preprocessar(colorida))


def test_qualquer_tamanho_vira_45x45():
    for altura, largura in [(20, 15), (45, 45), (300, 240), (100, 600)]:
        assert preprocessar(desenhar_sete(altura, largura, 3)).shape == (45, 45, 1)


def test_traco_grosso_fica_fino_como_no_dataset():
    """Caneta grossa (webcam) e traço fino (dataset) devem ficar parecidos."""
    fino = preprocessar(desenhar_sete(45, 45, 1))
    grosso = preprocessar(desenhar_sete(400, 400, 40, fundo=190, tinta=40))
    tinta_fino = (fino == 0).sum()
    tinta_grosso = (grosso == 0).sum()
    # Quantidade de tinta parecida (diferença de no máximo 40%)
    assert abs(tinta_grosso - tinta_fino) / tinta_fino < 0.4


def test_imagem_em_branco_nao_quebra():
    """Sem nenhum traço, devolve um quadrado todo branco em vez de dar erro."""
    saida = preprocessar(np.full((60, 60), 255, np.uint8))
    assert saida.shape == (45, 45, 1)
    assert (saida == 1.0).all()


def test_resultado_sempre_igual():
    """A mesma imagem tem que dar sempre a mesma saída (nada aleatório)."""
    img = desenhar_sete(120, 100, 8)
    assert np.array_equal(preprocessar(img), preprocessar(img))


def test_entrada_float_de_0_a_1():
    """Aceita também imagens já em 0..1 (como as que o Gradio pode entregar)."""
    img = desenhar_sete(100, 100, 6)
    assert np.array_equal(preprocessar(img), preprocessar(img / 255.0))


@pytest.mark.skipif(not PASTA_IMAGENS_BRUTAS.exists(), reason="dataset não está em data/raw/")
def test_imagens_reais_do_dataset():
    """Uma imagem de cada classe do dataset passa pela função sem erro e com traço."""
    for classe in CLASSES:
        pasta = PASTA_IMAGENS_BRUTAS / MAPA_PASTAS[classe]
        primeira = next(pasta.iterdir())
        saida = preprocessar(carregar_imagem(primeira))
        assert saida.shape == (45, 45, 1)
        assert (saida == 0.0).any(), f"classe {classe} ficou sem traço"
