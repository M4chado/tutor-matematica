"""Testes da voz (Etapa 8). Não tocam som: conferem o texto falado e que
a fala roda em segundo plano, sem travar a tela.

Rodar: python -m pytest
"""
import sys
import time
from types import SimpleNamespace

import pytest

from voz import Voz, escolher_voz, texto_para_fala


@pytest.mark.parametrize("texto, falado", [
    ("7 × 3 = 21", "7 vezes 3 é igual a 21"),
    ("12 + 21 = 33", "12 mais 21 é igual a 33"),
    ("x = 8 ÷ 2", "xis é igual a 8 dividido por 2"),
    ("2x = 8", "2 xis é igual a 8"),
    ("8 − 3 = 5", "8 menos 3 é igual a 5"),
    ("-3 + 5 = 2", "menos 3 mais 5 é igual a 2"),
    ("(8 - 3) × 4", "abre parênteses, 8 menos 3, fecha parênteses vezes 4"),
    ("7/2 = 3,5", "7 sobre 2 é igual a 3,5"),
    ("3 + 4 ≠ 8", "3 mais 4 é diferente de 8"),
    ("O resultado é **33**! 🎉", "O resultado é 33!"),
    ("Li assim: 12+?×3", "Li assim: 12 mais interrogação vezes 3"),
])
def test_contas_viram_palavras(texto, falado):
    assert texto_para_fala(texto) == falado


def test_palavras_normais_nao_mudam():
    texto = "Exemplo: a próxima (dica: tente devagar) - é fácil!"
    assert texto_para_fala(texto) == texto


def test_escolhe_a_voz_em_portugues():
    vozes = [SimpleNamespace(name="Microsoft Zira Desktop - English (United States)", languages=["en-US"]),
             SimpleNamespace(name="Microsoft Maria Desktop - Portuguese(Brazil)", languages=["pt-BR"])]
    assert escolher_voz(vozes).name.startswith("Microsoft Maria")
    assert escolher_voz(vozes[:1]) is None


def test_falar_nao_trava_e_pode_ser_interrompido():
    # Um "falante" de mentira que demora 10 s: falar() precisa voltar na hora
    voz = Voz(comando=[sys.executable, "-c", "import sys, time; sys.stdin.read(); time.sleep(10)"])
    inicio = time.perf_counter()
    voz.falar("um texto bem comprido")
    assert time.perf_counter() - inicio < 1.0
    assert voz.falando()
    voz.parar()
    assert not voz.falando()


def test_fala_nova_interrompe_a_anterior():
    voz = Voz(comando=[sys.executable, "-c", "import sys, time; sys.stdin.read(); time.sleep(10)"])
    voz.falar("primeira")
    primeira = voz.processo
    voz.falar("segunda")
    assert primeira.poll() is not None      # a primeira fala foi encerrada
    voz.parar()
