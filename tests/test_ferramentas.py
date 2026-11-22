"""Testes das ferramentas do agente (cálculo, validação, ambiguidade do x, exercícios).

Rodar: python -m pytest
"""
from pathlib import Path

import pytest
import sympy as sp

import ferramentas
from ferramentas import calcular, gerar_exercicio, resolver_ambiguidade_x, validar_expressao


def contas(resultado):
    return [p["conta"] for p in resultado["passos"]]


# ---------------------------------------------------------------------------
# calcular: expressões (Nível 1)
# ---------------------------------------------------------------------------

def test_ordem_das_operacoes_12_mais_7_vezes_3():
    r = calcular("12+7*3")
    assert r["ok"] and r["resultado"] == "33"
    assert contas(r) == ["12 + 7 × 3", "12 + 21", "33"]
    assert r["passos"][1]["operacao"] == "7 × 3 = 21"
    assert "antes de soma" in r["passos"][1]["regra"]


def test_parenteses_primeiro():
    r = calcular("(8-3)*4")
    assert r["resultado"] == "20"
    assert contas(r) == ["(8 - 3) × 4", "5 × 4", "20"]
    assert "parênteses" in r["passos"][1]["regra"]


def test_aceita_simbolos_da_cnn():
    """Lista de símbolos (como sai da segmentação), com × e ÷ de verdade."""
    assert calcular(["1", "2", "÷", "4", "+", "1"])["resultado"] == "4"
    assert calcular([{"simbolo": "6"}, {"simbolo": "×"}, {"simbolo": "7"}])["resultado"] == "42"


def test_multiplicacao_antes_mesmo_com_soma_na_esquerda():
    r = calcular("2+3+4×5")
    assert r["passos"][1]["operacao"] == "4 × 5 = 20"
    assert r["resultado"] == "25"


def test_esquerda_para_a_direita_na_subtracao():
    assert calcular("10-4-3")["resultado"] == "3"     # (10-4)-3, e não 10-(4-3)
    assert calcular("24÷4÷2")["resultado"] == "3"


def test_parenteses_aninhados():
    assert calcular("2×(3+(4-1))")["resultado"] == "12"


def test_multiplicacao_implicita_com_parenteses():
    assert calcular("2(3+4)")["resultado"] == "14"


def test_numeros_negativos():
    assert calcular("-3+5")["resultado"] == "2"
    assert calcular("(-2)×3")["resultado"] == "-6"
    r = calcular("10-(2-5)")
    assert r["resultado"] == "13"
    assert "10 - (-3)" in contas(r)


def test_divisao_nao_exata_vira_fracao_e_decimal():
    r = calcular("7÷2")
    assert r["resultado"] == "7/2" and r["decimal"] == "3,5"


def test_divisao_por_zero():
    r = calcular("8÷0")
    assert not r["ok"] and "zero" in r["erro"]
    assert not calcular("5÷(3-3)")["ok"]


# ---------------------------------------------------------------------------
# calcular: equações (Nível 2) e verificações
# ---------------------------------------------------------------------------

def test_equacao_2x_mais_3_igual_11():
    r = calcular("2x+3=11")
    assert r["ok"] and r["tipo"] == "equacao"
    assert r["resultado"] == "x = 4" and r["valor"] == "4"
    assert "2x = 11 - 3" in contas(r)
    assert "2x = 8" in contas(r)
    assert "x = 8 ÷ 2" in contas(r)
    assert r["passos"][-1]["regra"].endswith("✔")       # conferência bateu


@pytest.mark.parametrize("equacao, solucao", [
    ("3x=12", "4"),
    ("x+5=2", "-3"),
    ("2(x+1)=10", "4"),
    ("5x-2=3x+6", "4"),
    ("x÷2=3", "6"),
    ("4x=6", "3/2"),
    ("12=3x", "4"),
])
def test_varias_equacoes(equacao, solucao):
    r = calcular(equacao)
    assert r["ok"], r
    assert r["valor"] == solucao
    # Prova real: trocando o x pela solução, a igualdade tem que ser verdadeira
    substituida = equacao.replace("x", "(" + solucao.replace("/", "÷") + ")")
    assert calcular(substituida)["verdadeira"] is True


def test_equacao_sem_solucao_e_com_infinitas():
    assert calcular("x+1=x+2")["solucao"] == "nenhuma"
    assert calcular("2x=x+x")["solucao"] == "infinitas"


def test_equacao_fora_do_escopo():
    r = calcular("x×x=4")
    assert not r["ok"] and "grau" in r["erro"]
    assert not calcular("6÷x=2")["ok"]


def test_verificacao_verdadeira_e_falsa():
    assert calcular("3+4=7")["verdadeira"] is True
    r = calcular("3+4=8")
    assert r["verdadeira"] is False and r["resultado"] == "falsa"
    assert calcular("2×3=1+5")["verdadeira"] is True


# ---------------------------------------------------------------------------
# validar_expressao
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("conta", ["12+7×3", "(8-3)×4", "-3+5", "2x+3=11", "x=-3", "2(3+4)", "3+4=7"])
def test_contas_validas(conta):
    assert validar_expressao(conta)["valida"], validar_expressao(conta)


@pytest.mark.parametrize("conta, trecho", [
    ("12++3", "Dois operadores seguidos"),
    ("3×÷2", "Dois operadores seguidos"),
    ("(8-3", "Falta fechar"),
    ("8-3)", "não foi aberto"),
    ("+3", "não pode começar"),
    ("3+", "não pode terminar"),
    ("", "vazia"),
    ("()", "vazios"),
    ("2x+3", "não tem '='"),
    ("1=2=3", "mais de um"),
    ("=5", "vazio"),
    ("(3)4", "sem operador"),
])
def test_contas_invalidas_explicam_o_problema(conta, trecho):
    r = validar_expressao(conta)
    assert not r["valida"]
    assert any(trecho in p for p in r["problemas"]), r["problemas"]


def test_calcular_recusa_conta_invalida_com_explicacao():
    r = calcular("12++3")
    assert not r["ok"] and "operadores" in r["erro"]


# ---------------------------------------------------------------------------
# resolver_ambiguidade_x
# ---------------------------------------------------------------------------

def test_x_entre_numeros_e_multiplicacao():
    assert resolver_ambiguidade_x(["3", "x", "4"])["expressao"] == "3×4"


def test_x_com_igual_e_incognita():
    r = resolver_ambiguidade_x(list("2×+3=11"))   # a CNN leu × no lugar do x
    assert r["expressao"] == "2x+3=11"
    assert r["decisoes"][0]["mudou"] is True


def test_x_entre_numeros_numa_verificacao_e_multiplicacao():
    assert resolver_ambiguidade_x(list("3x4=12"))["expressao"] == "3×4=12"


def test_x_antes_do_igual_e_incognita():
    assert resolver_ambiguidade_x(list("3×=12"))["expressao"] == "3x=12"
    assert resolver_ambiguidade_x(list("x+1=3"))["expressao"] == "x+1=3"


def test_x_depois_de_parenteses_e_multiplicacao():
    assert resolver_ambiguidade_x(list("(2+1)x4"))["expressao"] == "(2+1)×4"


def test_conta_lida_pela_cnn_ate_o_resultado():
    """Fluxo completo: a CNN leu 12+7x3 (com x); o contexto decide e o SymPy calcula."""
    simbolos = resolver_ambiguidade_x(list("12+7x3"))["simbolos"]
    assert calcular(simbolos)["resultado"] == "33"


# ---------------------------------------------------------------------------
# gerar_exercicio
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("original", ["12+7×3", "(8-3)×4", "8÷4", "2x+3=11", "3x=12", "5x-2=3x+6"])
def test_exercicio_parecido_e_resposta_certa(original):
    r = gerar_exercicio(original, semente=1)
    assert r["ok"], r
    assert r["simbolos"] != original
    # mesma "forma": mesmos operadores, parênteses e x, na mesma ordem
    forma = lambda s: [c for c in s if not c.isdigit()]
    assert forma(r["simbolos"]) == forma(original.replace("*", "×"))
    # a resposta informada é a mesma que a ferramenta calcula
    assert calcular(r["simbolos"])["resultado"] == r["resposta"]


def test_exercicio_de_equacao_tem_x_inteiro_e_pequeno():
    for semente in range(10):
        r = gerar_exercicio("2x+3=11", semente=semente)
        x = sp.Rational(calcular(r["simbolos"])["valor"])
        assert x.q == 1 and 1 <= x <= 12


def test_exercicio_troca_numeros_de_1_digito_por_2_a_9():
    """Em 12+7×3, o 7 e o 3 viram números de 2 a 9 (nunca 0 nem 1)."""
    for semente in range(20):
        r = gerar_exercicio("12+7×3", semente=semente)
        numeros = [t for t in ferramentas._tokenizar(list(r["simbolos"])) if isinstance(t, int)]
        assert len(numeros) == 3
        assert 10 <= numeros[0] <= 99
        assert all(2 <= n <= 9 for n in numeros[1:])


def test_ultima_operacao_tem_regra_certa():
    assert calcular("12+7×3")["passos"][-1]["regra"] == "só falta esta operação"


def test_exercicio_de_divisao_e_exato():
    for semente in range(10):
        r = gerar_exercicio("12÷4", semente=semente)
        assert "/" not in r["resposta"]


def test_exercicio_com_mesma_semente_e_igual():
    assert gerar_exercicio("12+7×3", semente=7) == gerar_exercicio("12+7×3", semente=7)


def test_exercicio_de_verificacao_pratica_o_lado_esquerdo():
    r = gerar_exercicio("3+4=7", semente=1)
    assert r["ok"] and "=" not in r["simbolos"]


# ---------------------------------------------------------------------------
# Segurança
# ---------------------------------------------------------------------------

def test_codigo_nao_usa_eval():
    """Regra do projeto: nada de eval() (nem parse_expr/sympify, que usam eval por dentro).

    Analisa a árvore do código Python (não o texto), então comentários e
    explicações que citam essas palavras não contam."""
    import ast

    arvore = ast.parse(Path(ferramentas.__file__).read_text(encoding="utf-8"))
    usados = {n.id for n in ast.walk(arvore) if isinstance(n, ast.Name)}
    usados |= {n.attr for n in ast.walk(arvore) if isinstance(n, ast.Attribute)}
    assert not usados & {"eval", "exec", "parse_expr", "sympify", "compile"}


def test_texto_malicioso_e_recusado():
    r = calcular("__import__('os')")
    assert not r["ok"] and "não conheço" in r["erro"]
