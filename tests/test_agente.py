"""Testes do agente (Etapa 6) com um LLM FALSO, roteirizado.

Por que um LLM falso? Para provar que as regras do CÓDIGO funcionam mesmo
quando o LLM erra ou desobedece (esquece a ferramenta, escreve conta errada,
revela a resposta...). E os testes rodam em segundos, sem o Ollama.

Rodar: python -m pytest
"""
from pathlib import Path

import pytest

import agente as ag
from agente import (Agente, conferir_contas_do_texto, interpretar_resposta, limpar_texto, opcoes_da_duvida,
                    simbolos_simulados)


class LLMFalso:
    """Devolve respostas prontas, na ordem. Cada uma é um texto ou uma chamada de ferramenta."""

    def __init__(self, *roteiro):
        self.roteiro = list(roteiro)
        self.recebidas = []   # mensagens que o agente mandou em cada chamada

    def __call__(self, mensagens, ferramentas):
        self.recebidas.append(mensagens)
        if not self.roteiro:
            raise AssertionError("O agente chamou o LLM mais vezes do que o esperado")
        passo = self.roteiro.pop(0)
        if isinstance(passo, Exception):
            raise passo
        if isinstance(passo, tuple):          # ("calcular", {"expressao": "..."})
            return {"texto": "", "chamadas": [{"nome": passo[0], "argumentos": passo[1]}]}
        return {"texto": passo, "chamadas": []}


BOA_33 = "Primeiro a multiplicação: 7 × 3 = 21. A conta fica 12 + 21. Agora somamos: 12 + 21 = 33. O resultado é 33!"


def conta(texto, *duvidas):
    return simbolos_simulados(texto, duvidas)


# ---------------------------------------------------------------------------
# Decidir quando perguntar
# ---------------------------------------------------------------------------

def test_confianca_alta_nao_pergunta():
    assert opcoes_da_duvida({"simbolo": "7", "confianca": 0.95, "top3": [("7", 0.95)]}, 0.8) is None


def test_confianca_baixa_pergunta_com_as_opcoes_da_cnn():
    s = {"simbolo": "7", "confianca": 0.6, "top3": [("7", 0.6), ("1", 0.35), ("4", 0.05)]}
    assert opcoes_da_duvida(s, 0.8) == ["7", "1"]       # o 4 (5%) não é plausível


def test_duvida_so_entre_x_e_vezes_nao_pergunta():
    """O contexto resolve x/× (Etapa 5), então não precisa incomodar o visitante."""
    s = {"simbolo": "×", "confianca": 0.55, "top3": [("×", 0.55), ("x", 0.43), ("4", 0.02)]}
    assert opcoes_da_duvida(s, 0.8) is None


def test_pergunta_antes_de_calcular_sem_chamar_o_llm():
    llm = LLMFalso()                                     # se o LLM for chamado, o teste falha
    r = Agente(chat=llm).ler_conta(conta("12+7*3", "4=1"))
    assert r["tipo"] == "pergunta"
    assert "4º símbolo" in r["texto"] and "7 ou 1" in r["texto"]
    assert llm.recebidas == []


def test_depois_da_resposta_calcula_com_o_simbolo_escolhido():
    llm = LLMFalso(("calcular", {"expressao": "12+1×3"}),
                   "Primeiro a multiplicação: 1 × 3 = 3. Depois 12 + 3 = 15. O resultado é 15!")
    agente = Agente(chat=llm)
    agente.ler_conta(conta("12+7*3", "4=1"))
    assert agente.responder("não sei")["tipo"] == "pergunta"   # não entendeu: pergunta de novo
    r = agente.responder("é o 1")
    assert r["calculo"]["resultado"] == "15"
    assert r["verificacao"]["aprovada"]


@pytest.mark.parametrize("texto, opcoes, esperado", [
    ("é o 7", ["7", "1"], "7"), ("sim", ["7", "1"], "7"), ("o segundo", ["7", "1"], "1"),
    ("vezes", ["×", "4"], "×"), ("xis", ["×", "4"], "×"), ("é um 1", ["7", "1"], "1"),
    ("4", ["7", "1"], "4"), ("sei lá", ["7", "1"], None),
])
def test_entende_a_resposta_do_visitante(texto, opcoes, esperado):
    assert interpretar_resposta(texto, opcoes) == esperado


# ---------------------------------------------------------------------------
# Ferramentas e travas
# ---------------------------------------------------------------------------

def test_fluxo_normal_llm_chama_calcular_e_explica():
    llm = LLMFalso(("calcular", {"expressao": "12+7×3"}), BOA_33)
    r = Agente(chat=llm).ler_conta(conta("12+7*3"))
    assert r["texto"] == BOA_33
    assert r["ferramentas"] == ["calcular"]
    assert r["verificacao"] == {"aprovada": True, "tentativas": 1, "problemas": [], "reprovadas": [], "aviso": None}
    # o resultado da ferramenta foi entregue ao LLM
    assert any(m["role"] == "tool" and '"resultado": "33"' in m["content"] for m in llm.recebidas[1])


def test_llm_esquece_a_ferramenta_e_o_codigo_calcula():
    llm = LLMFalso("Acho que dá 33.", BOA_33)
    r = Agente(chat=llm).ler_conta(conta("12+7*3"))
    assert "calcular (forçado pelo código)" in r["ferramentas"]
    assert r["calculo"]["resultado"] == "33"


def test_llm_nao_consegue_trocar_a_conta_lida():
    """Na conta nova, o código calcula exatamente o que a CNN leu, mesmo se o LLM pedir outra."""
    llm = LLMFalso(("calcular", {"expressao": "12+8×3"}), BOA_33)
    r = Agente(chat=llm).ler_conta(conta("12+7*3"))
    assert r["calculo"]["expressao"] == "12 + 7 × 3"


def test_x_lido_como_vezes_vira_incognita_para_o_llm():
    llm = LLMFalso(("calcular", {"expressao": "2x+3=11"}), "Passamos o 3: 2x = 8, então x = 8 ÷ 2 = 4. O x vale 4.")
    r = Agente(chat=llm).ler_conta(conta("2*+3=11"))       # a CNN leu × no lugar do x
    assert r["expressao"] == "2x+3=11"
    assert r["calculo"]["resultado"] == "x = 4"


def test_conta_mal_escrita_explica_sem_chamar_o_llm():
    llm = LLMFalso()
    r = Agente(chat=llm).ler_conta(conta("12++3"))
    assert r["tipo"] == "erro" and "operadores seguidos" in r["texto"]
    assert llm.recebidas == []


# ---------------------------------------------------------------------------
# Chain of Verification
# ---------------------------------------------------------------------------

def test_conferencia_pega_conta_errada_e_pede_de_novo():
    errada = "Primeiro 7 × 3 = 20. Depois 12 + 20 = 32. O resultado é 32!"
    llm = LLMFalso(("calcular", {"expressao": "12+7×3"}), errada, BOA_33)
    r = Agente(chat=llm).ler_conta(conta("12+7*3"))
    assert r["texto"] == BOA_33
    assert r["verificacao"]["tentativas"] == 2 and r["verificacao"]["reprovadas"] == [errada]
    assert any("7 × 3 = 20" in p for p in r["verificacao"]["problemas"])
    # o LLM recebeu o motivo da reprovação
    assert "Revisão automática" in llm.recebidas[-1][-1]["content"]


def test_llm_sempre_errado_usa_explicacao_automatica():
    errada = "7 × 3 = 20, então dá 32."
    llm = LLMFalso(("calcular", {"expressao": "12+7×3"}), errada, errada, errada)
    r = Agente(chat=llm).ler_conta(conta("12+7*3"))
    assert not r["verificacao"]["aprovada"]
    assert "**33**" in r["texto"] and "32" not in r["texto"]


def test_ollama_desligado_ainda_responde_certo():
    llm = LLMFalso(ConnectionError("Ollama fechado"))
    r = Agente(chat=llm).ler_conta(conta("(8-3)*4"))
    assert "**20**" in r["texto"]
    assert "indisponível" in r["verificacao"]["aviso"]


def test_conferir_contas_do_texto():
    assert conferir_contas_do_texto(BOA_33) == []
    assert conferir_contas_do_texto("x = 8 ÷ 2 = 4 e 7 ÷ 2 = 3,5 e 7 x 3 = 21 e 10 − (2 − 5) = 13") == []
    assert len(conferir_contas_do_texto("5 × 4 = 25")) == 1
    assert len(conferir_contas_do_texto("12 + 7 × 3 = 12 + 21 = 34")) == 1


def test_limpa_latex():
    assert limpar_texto(r"Explicação: \(7 \times 3 = 21\) e \(\frac{7}{2}\)") == "7 × 3 = 21 e 7/2"


# ---------------------------------------------------------------------------
# Exercícios e histórico
# ---------------------------------------------------------------------------

def _agente_com_exercicio(*roteiro_extra):
    llm = LLMFalso(("calcular", {"expressao": "12+7×3"}), BOA_33,
                   ("gerar_exercicio", {"expressao": "12+7×3"}), "Tente resolver este exercício!", *roteiro_extra)
    agente = Agente(chat=llm)
    agente.ler_conta(conta("12+7*3"))
    r = agente.responder("me dá um exercício")
    return agente, llm, r


def test_exercicio_fica_guardado_sem_a_resposta_para_o_llm():
    agente, llm, r = _agente_com_exercicio()
    assert r["tipo"] == "exercicio" and agente.exercicio
    resultado_ferramenta = [m for m in llm.recebidas[-1] if m["role"] == "tool"][-1]["content"]
    assert agente.exercicio["resposta"] not in resultado_ferramenta.replace(agente.exercicio["exercicio"], "")


def test_resposta_do_exercicio_e_conferida_pelo_codigo():
    agente, llm, _ = _agente_com_exercicio()
    exercicio, resposta = agente.exercicio["exercicio"], agente.exercicio["resposta"]
    llm.roteiro += [("calcular", {"expressao": exercicio}), f"Parabéns, você acertou! O resultado é {resposta}."]
    r = agente.responder(f"dá {resposta}")
    assert r["veredito"]["acertou"] is True
    assert r["verificacao"]["aprovada"]
    assert agente.exercicio is None                          # exercício fechado


def test_llm_nao_pode_dizer_que_acertou_quem_errou():
    agente, llm, _ = _agente_com_exercicio()
    exercicio, resposta = agente.exercicio["exercicio"], agente.exercicio["resposta"]
    llm.roteiro += [("calcular", {"expressao": exercicio}), f"Parabéns, você acertou! Dá {resposta}.",
                    f"Não foi dessa vez. O resultado é {resposta}."]
    r = agente.responder("dá 1")
    assert r["veredito"]["acertou"] is False
    assert r["texto"].startswith("Não foi dessa vez")


def test_historico_guarda_a_conversa():
    llm = LLMFalso(("calcular", {"expressao": "12+7×3"}), BOA_33, "Porque a multiplicação vem antes da soma.")
    agente = Agente(chat=llm)
    agente.ler_conta(conta("12+7*3"))
    agente.responder("por que multiplicou primeiro?")
    ultima_chamada = llm.recebidas[-1]
    assert ultima_chamada[0]["role"] == "system"
    assert any(m["content"] == BOA_33 for m in ultima_chamada)   # o LLM lembra da explicação anterior


# ---------------------------------------------------------------------------
# Honestidade da calibração
# ---------------------------------------------------------------------------

def test_calibracao_usa_validacao_e_nunca_o_teste():
    codigo = (Path(ag.__file__).parent / "calibrar_confianca.py").read_text(encoding="utf-8")
    assert "validacao.npz" in codigo and "teste.npz" not in codigo
