"""Etapa 6: o agente com LLM local (Ollama + qwen3:4b-instruct).

O que torna isto um AGENTE (e não só um chatbot):
  1. PERCEBE:   recebe a conta lida pela CNN, com a confiança de cada símbolo.
  2. DECIDE:    se algum símbolo está duvidoso, pergunta antes; senão, calcula.
  3. USA FERRAMENTAS: o LLM chama calcular / gerar_exercicio / perguntar_ao_visitante
               (tool calling). Ele NUNCA faz contas sozinho.
  4. AGE E CONFERE: explica a conta e, antes de responder, confere com o SymPy
               cada número que escreveu (Chain of Verification). Se errou, refaz.

Regras que ficam no CÓDIGO (e não só no prompt, porque um LLM pode desobedecer):
  - Símbolo com confiança abaixo de config.LIMITE_CONFIANCA -> pergunta antes de calcular.
  - Conta nova sem chamada de calcular -> o código chama a ferramenta mesmo assim.
  - Explicação com número errado -> refaz (até 3 tentativas); se não acertar,
    usa a explicação automática, montada direto dos passos do SymPy.

Teste pelo terminal (com o Ollama aberto):
  python src/agente.py "12+7*3"
  python src/agente.py "12+7*3" --duvida 4=1      (4º símbolo duvidoso: 7 ou 1?)
  python src/agente.py --imagem foto.jpg           (lê a conta de uma foto com a CNN)
"""
import argparse
import json
import re
import sys
import time

import sympy as sp

from config import CLASSES, LIMITE_CONFIANCA, MODELO_LLM
from ferramentas import (CRUZES, EQUIVALENTES, calcular, decimal_br, gerar_exercicio,
                         resolver_ambiguidade_x, validar_expressao)

# ---------------------------------------------------------------------------
# Ajustes
# ---------------------------------------------------------------------------
PROB_ALTERNATIVA = 0.10      # um 2º palpite só vira opção na pergunta se tiver >= 10%
MAX_TENTATIVAS = 3           # explicações que o LLM pode tentar antes do plano B
MAX_RODADAS_FERRAMENTAS = 5  # evita o LLM ficar chamando ferramentas sem parar
MAX_MENSAGENS = 24           # histórico guardado (mensagens antigas saem, para caber na memória)
OPCOES_LLM = {"temperature": 0.2, "num_ctx": 4096, "num_predict": 350}
MANTER_CARREGADO = "30m"     # o Ollama mantém o modelo na memória por 30 min sem uso

# ---------------------------------------------------------------------------
# Prompt de sistema: persona + regras + exemplos de boas explicações (few-shot)
# ---------------------------------------------------------------------------
PROMPT_SISTEMA = """Você é o Tutor, um professor de matemática paciente e animado que conversa com visitantes de uma feira (crianças e adultos).

Regras:
1. Você NUNCA faz contas de cabeça. Para qualquer conta, chame a ferramenta calcular e use só os números que ela devolver.
2. Responda em português do Brasil, com frases curtas e simples, em no máximo 6 linhas.
3. Explique seguindo os passos da ferramenta, na ordem: parênteses primeiro, depois × e ÷, depois + e −, da esquerda para a direita. Diga o porquê de cada passo.
4. Escreva as contas com os símbolos × e ÷. Não use LaTeX, nem $, nem \\( \\).
5. Termine dizendo o resultado final.
6. Se o visitante pedir um exercício, chame gerar_exercicio e NÃO conte a resposta: peça que ele tente. Quando ele responder, chame calcular com o exercício para conferir.
7. Se não entender o que o visitante quis dizer, chame perguntar_ao_visitante.

Exemplos de boas explicações:

Conta 12 + 7 × 3 (ferramenta: 12 + 21, depois 33)
Explicação: Primeiro fazemos a multiplicação, porque ela vem antes da soma: 7 × 3 = 21. A conta fica 12 + 21. Agora somamos: 12 + 21 = 33. O resultado é 33!

Conta (8 − 3) × 4 (ferramenta: 5 × 4, depois 20)
Explicação: O que está dentro dos parênteses vem primeiro: 8 − 3 = 5. A conta fica 5 × 4. Agora é só multiplicar: 5 × 4 = 20. O resultado é 20!

Conta 2x + 3 = 11 (ferramenta: 2x = 11 − 3, 2x = 8, x = 8 ÷ 2, x = 4)
Explicação: Queremos descobrir o x. Passamos o 3 para o outro lado fazendo a operação contrária: 2x = 11 − 3, ou seja, 2x = 8. Se 2 vezes o x dá 8, dividimos por 2: x = 8 ÷ 2 = 4. Conferindo: 2 × 4 + 3 = 11. Certo! O x vale 4."""

# Descrição das ferramentas no formato que o Ollama entende (tool calling)
FERRAMENTAS_LLM = [
    {"type": "function", "function": {
        "name": "calcular",
        "description": "Resolve uma conta ou equação do 1º grau, passo a passo, com o resultado exato. "
                       "Use SEMPRE que houver qualquer conta.",
        "parameters": {"type": "object", "required": ["expressao"], "properties": {
            "expressao": {"type": "string", "description": "a conta, por exemplo 12+7×3 ou 2x+3=11"}}}}},
    {"type": "function", "function": {
        "name": "gerar_exercicio",
        "description": "Cria um exercício parecido com uma conta, com outros números, para o visitante treinar.",
        "parameters": {"type": "object", "required": ["expressao"], "properties": {
            "expressao": {"type": "string", "description": "a conta que serve de modelo"}}}}},
    {"type": "function", "function": {
        "name": "perguntar_ao_visitante",
        "description": "Faz uma pergunta ao visitante quando você não tem certeza de um símbolo ou do que ele quis dizer.",
        "parameters": {"type": "object", "required": ["simbolo", "opcoes"], "properties": {
            "simbolo": {"type": "string", "description": "o símbolo ou a frase em dúvida"},
            "opcoes": {"type": "array", "items": {"type": "string"}, "description": "as opções possíveis"}}}}},
]

# Nomes amigáveis para a pergunta ("é × (vezes) ou 4?")
NOMES = {"×": "× (vezes)", "x": "x (a incógnita)", "÷": "÷ (dividir)", "+": "+ (mais)", "-": "- (menos)",
         "=": "= (igual)", "(": "( (abre parênteses)", ")": ") (fecha parênteses)"}
# Palavras que o visitante pode usar para responder
PALAVRAS = {"zero": "0", "dois": "2", "três": "3", "tres": "3", "quatro": "4", "cinco": "5", "seis": "6",
            "sete": "7", "oito": "8", "nove": "9", "mais": "+", "soma": "+", "menos": "-", "vezes": "×",
            "multiplicação": "×", "multiplicacao": "×", "xis": "x", "incógnita": "x", "incognita": "x",
            "dividido": "÷", "divisão": "÷", "divisao": "÷", "igual": "=", "abre": "(", "fecha": ")"}
SIM = {"sim", "s", "isso", "exato", "certo", "correto", "é", "e"}
ORDINAIS = {"primeiro": 0, "primeira": 0, "segundo": 1, "segunda": 1, "terceiro": 2, "terceira": 2}
# Respostas que dizem "isso não é símbolo nenhum" (sujeira, objeto ao lado da folha)
REMOVER = "remover"
NAO_E_SIMBOLO = re.compile(r"\b(nada|nenhum|nenhuma|sujeira|mancha|apag\w*|tir[ae]\w*|remov\w*)\b"
                           r"|n[ãa]o (tem|existe|há|ha)\b")


# ---------------------------------------------------------------------------
# 1. Decidir se é preciso perguntar (regra no código)
# ---------------------------------------------------------------------------

def opcoes_da_duvida(simbolo, limite=LIMITE_CONFIANCA):
    """Devolve as opções para perguntar ao visitante, ou None se não há dúvida.

    Há dúvida quando a confiança da CNN está abaixo do limite. Exceção: se as
    opções plausíveis são só x e ×, não perguntamos, porque o contexto decide
    (ferramenta resolver_ambiguidade_x da Etapa 5)."""
    if simbolo["confianca"] >= limite:
        return None
    top3 = simbolo.get("top3") or [(simbolo["simbolo"], simbolo["confianca"])]
    plausiveis = [simbolo["simbolo"]] + [s for s, p in top3 if p >= PROB_ALTERNATIVA and s != simbolo["simbolo"]]
    if set(plausiveis) <= CRUZES:
        return None
    opcoes = list(plausiveis)
    for s, _ in top3:                       # garante pelo menos 2 opções
        if len(opcoes) >= 2:
            break
        if s not in opcoes:
            opcoes.append(s)
    # x e × juntos viram uma opção só (o contexto decide qual dos dois é)
    if "x" in opcoes and "×" in opcoes:
        opcoes.remove("x" if opcoes.index("×") < opcoes.index("x") else "×")
    return opcoes


def perguntar_ao_visitante(simbolo, opcoes, posicao=None, leitura=None):
    """Ferramenta: monta a pergunta para o visitante (o agente para e espera a resposta)."""
    nomes = [NOMES.get(o, o) for o in opcoes]
    escolha = nomes[0] if len(nomes) == 1 else ", ".join(nomes[:-1]) + " ou " + nomes[-1]
    if posicao is not None:
        texto = f"Li assim: **{leitura}**\n\nNão tenho certeza do {posicao}º símbolo. Ele é {escolha}?"
        if len(opcoes) == 1:
            texto += " (responda sim ou escreva o símbolo certo)"
        texto += " Se não for símbolo nenhum (sujeira), responda **nenhum**."
    else:
        texto = f"Fiquei em dúvida sobre \"{simbolo}\". Você quis dizer {escolha}?"
    return {"pergunta": texto, "simbolo": simbolo, "opcoes": list(opcoes)}


def interpretar_resposta(texto, opcoes):
    """Entende a resposta do visitante a uma pergunta. Ex.: "é o 7", "sim", "vezes".
    Devolve o símbolo escolhido, REMOVER ("não é símbolo nenhum") ou None se
    não deu para entender."""
    t = texto.strip().lower()
    if NAO_E_SIMBOLO.search(t):
        return REMOVER
    palavras = re.findall(r"\d|[a-zà-ú]+|[^\sa-zà-ú\d]", t)
    if not palavras:
        return None
    if palavras[0] in SIM and len(palavras) <= 2 and not any(p in PALAVRAS or p.isdigit() for p in palavras[1:]):
        return opcoes[0]
    if t in {"não", "nao", "n"} and len(opcoes) == 2:
        return opcoes[1]
    for p in palavras:
        if p in ORDINAIS and ORDINAIS[p] < len(opcoes):
            return opcoes[ORDINAIS[p]]

    candidatos = []
    for p in palavras:
        s = EQUIVALENTES.get(p, p)
        s = PALAVRAS.get(s, s)
        if s in CLASSES and s not in candidatos:
            candidatos.append(s)
    if not candidatos and ("um" in palavras or "uma" in palavras):
        candidatos = ["1"]                  # "um" só conta como número se não houver outra coisa

    def na_lista(c):                        # x e × valem como a mesma opção
        if c in opcoes:
            return c
        if c in CRUZES:
            return next((o for o in opcoes if o in CRUZES), None)
        return None

    escolhidos = list(dict.fromkeys(filter(None, map(na_lista, candidatos))))
    if len(escolhidos) == 1:
        return escolhidos[0]
    if not escolhidos and len(candidatos) == 1:   # o visitante corrigiu para outro símbolo
        return candidatos[0]
    return None


# ---------------------------------------------------------------------------
# 2. Chain of Verification: conferir com o SymPy os números da explicação
# ---------------------------------------------------------------------------

def limpar_texto(texto):
    """Tira LaTeX e marcas que o modelo às vezes usa, e padroniza os símbolos."""
    texto = re.sub(r"<think>.*?</think>", "", texto or "", flags=re.S)
    texto = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", texto)
    for antigo, novo in ((r"\times", "×"), (r"\cdot", "×"), (r"\div", "÷"), (r"\(", ""), (r"\)", ""),
                         (r"\[", ""), (r"\]", ""), ("$", ""), (r"\,", " ")):
        texto = texto.replace(antigo, novo)
    texto = re.sub(r"\\text\{([^}]*)\}", r"\1", texto)
    texto = re.sub(r"^\s*\**explica[çc][ãa]o\**\s*:\s*\**", "", texto, flags=re.I)  # rótulo copiado dos exemplos
    return texto.strip()


def _para_conferir(texto):
    """Prepara o texto para achar contas: − vira -, 7 x 3 vira 7 × 3, 3,5 vira (35÷10)."""
    t = texto.replace("−", "-").replace("–", "-").replace("·", "×").replace("*", "×")
    t = re.sub(r"(\d)\s*[xX]\s*(?=\d)", r"\1 × ", t)
    t = re.sub(r"(\d+),(\d+)", lambda m: f"({m.group(1)}{m.group(2)}÷{10 ** len(m.group(2))})", t)
    return t


def _limpar_segmento(seg):
    seg = seg.strip().strip(",.;:").strip()
    while seg and seg[-1] in "+-×÷/":      # sobra de pontuação no fim
        seg = seg[:-1].strip()
    while seg.startswith("(") and seg.count("(") > seg.count(")"):
        seg = seg[1:].strip()
    while seg.endswith(")") and seg.count(")") > seg.count("("):
        seg = seg[:-1].strip()
    return seg


def _valor(seg):
    """Valor exato de um pedaço de conta (só números e operações), ou None.
    Pedaço com a incógnita (9x - 18) não tem valor fixo: não dá para conferir."""
    if not seg or not re.search(r"\d", seg) or re.search(r"[xX]", seg):
        return None
    r = calcular(seg)
    return sp.Rational(r["valor"]) if r["ok"] and r["tipo"] == "expressao" else None


def conferir_contas_do_texto(texto):
    """Acha cada "conta = resultado" escrita no texto e confere com o SymPy.

    Ex.: "7 × 3 = 21" ✔ ; "7 × 3 = 20" ✘. Devolve a lista de contas erradas."""
    erradas = []
    # pedaços do texto formados só por números, operações, parênteses, "=" e a incógnita x.
    # O x entra no pedaço para "9x - 18 = 18" não virar "- 18 = 18" (que pareceria errado);
    # o x de palavras ("exemplo") não conta, porque está grudado em outras letras.
    for trecho in re.findall(r"(?:[\d+\-×÷/()=\s,.]|(?<![A-Za-zÀ-ÿ])[xX](?![A-Za-zÀ-ÿ]))+", _para_conferir(texto)):
        if "=" not in trecho:
            continue
        segmentos = [_limpar_segmento(s) for s in trecho.split("=")]
        for a, b in zip(segmentos, segmentos[1:]):
            va, vb = _valor(a), _valor(b)
            if va is not None and vb is not None and va != vb:
                erradas.append(f"{a} = {b} (o certo é {a} = {sp.nsimplify(va)})")
    return erradas


def _numero_no_texto(numero, texto):
    """O número aparece sozinho no texto? ("33" em "dá 33!" sim; em "133" não)."""
    return re.search(rf"(?<![\d/]){re.escape(numero)}(?![\d/]|,\d)", texto) is not None


def resultado_mencionado(calculo, texto):
    """A explicação diz o resultado que a ferramenta calculou?"""
    t = _para_conferir(texto).lower()
    texto_n = texto.replace("−", "-").lower()
    if calculo["tipo"] == "verificacao":
        if calculo["verdadeira"]:
            return re.search(r"verdadeir|corret|cert|igua", t) is not None
        return re.search(r"fals|errad|incorret|diferente|não (está|é) (cert|igual)|≠", t) is not None
    if calculo["tipo"] == "equacao" and calculo["solucao"] != "unica":
        if calculo["solucao"] == "nenhuma":
            return re.search(r"nenhum|não tem solução|sem solução|impossível|não existe", t) is not None
        return re.search(r"qualquer|infinit|todos", t) is not None
    valor = sp.Rational(calculo["valor"])
    formas = {calculo["valor"]}
    if valor.q != 1:
        formas |= {decimal_br(valor), f"{valor.p} ÷ {valor.q}", f"{valor.p}÷{valor.q}"}
    return any(_numero_no_texto(f, texto_n) for f in formas)


NUMERO = r"\d+(?:,\d+)?"   # 12, 3,5


def numeros_das_contas(texto):
    """Números que aparecem dentro de contas escritas no texto ("7 × 3 = 21" -> 7, 3, 21).
    Decimais ficam inteiros: "3,5" é um número só."""
    t = re.sub(r"(\d)\s*[xX]\s*(?=\d)", r"\1 × ", texto.replace("−", "-").replace("–", "-").replace("*", "×"))
    numeros = set()
    for trecho in re.findall(r"[\d+\-×÷/()=\s,.]+", t):
        if "=" in trecho and re.search(r"\d\s*\)?\s*[+\-×÷/]\s*\(?\s*\d", trecho):
            numeros |= set(re.findall(NUMERO, trecho))
    return numeros


def problemas_da_explicacao(texto, calculo, exercicio=None, veredito=None, conhecidos=None):
    """Chain of Verification: lista o que está errado na explicação (vazia = aprovada).

    exercicio:  exercício criado neste turno (a resposta não pode aparecer).
    veredito:   conferência da resposta do visitante feita pelo código (acertou ou não).
    conhecidos: texto com tudo de onde o LLM pode tirar números (ferramentas e o que o
                visitante escreveu). Número de conta fora disso foi INVENTADO, mesmo
                que a conta inventada esteja certa (o LLM fez conta de cabeça)."""
    problemas = [f"A conta {c} está errada." for c in conferir_contas_do_texto(texto)]
    if conhecidos is not None:
        inventados = numeros_das_contas(texto) - set(re.findall(NUMERO, conhecidos))
        if inventados:
            problemas.append("A explicação tem contas com números que não vieram da ferramenta calcular ("
                             + ", ".join(sorted(inventados)) + "). Não faça contas de cabeça: "
                             "chame calcular com a conta do visitante.")
    if veredito:
        t = texto.lower()
        elogio = re.search(r"parab[ée]ns|(?<!não )acertou|(?<!não )está cert|muito bem|isso mesmo", t)
        if veredito["acertou"] and not elogio:
            problemas.append("O visitante ACERTOU o exercício; diga isso a ele.")
        if not veredito["acertou"] and elogio:
            problemas.append("O visitante ERROU o exercício; não diga que ele acertou.")
    if calculo and calculo.get("ok"):
        if not resultado_mencionado(calculo, texto):
            problemas.append(f"A explicação não diz o resultado certo, que é {calculo['resultado']}.")
        if calculo["tipo"] == "equacao" and calculo["solucao"] == "unica":
            for x in re.findall(r"\bx\s*=\s*(-?\d+(?:/\d+)?)(?!\s*[÷×+\-/\d])", _para_conferir(texto)):
                if sp.Rational(x) != sp.Rational(calculo["valor"]):
                    problemas.append(f"O texto diz x = {x}, mas o certo é x = {calculo['valor']}.")
    if exercicio and re.search(rf"(=|é|dá|resposta[^.\d]*)\s*\**{re.escape(exercicio['resposta'])}\b", texto):
        problemas.append("A explicação revelou a resposta do exercício; não conte a resposta.")
    return problemas


def explicacao_automatica(calculo):
    """Plano B, sem LLM: explicação montada direto dos passos do SymPy (sempre certa)."""
    if not calculo.get("ok"):
        return f"Não consegui resolver essa conta: {calculo['erro']}"
    linhas = [f"Vamos resolver **{calculo['expressao']}**:"]
    for passo in calculo["passos"][1:]:
        if passo["operacao"]:
            linhas.append(f"- {passo['operacao']} ({passo['regra']}), a conta fica **{passo['conta']}**")
        else:
            linhas.append(f"- **{passo['conta']}** ({passo['regra']})")
    linhas.append(f"Resultado: **{calculo['resultado']}**")
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# 3. Conversa com o Ollama
# ---------------------------------------------------------------------------

def _resumo_para_llm(calculo):
    """Resultado do calcular em formato curto (menos texto = resposta mais rápida)."""
    if not calculo["ok"]:
        return {"ok": False, "erro": calculo["erro"]}
    passos = []
    for p in calculo["passos"][1:]:
        detalhe = f"{p['operacao']}: {p['regra']}" if p["operacao"] else p["regra"]
        passos.append(f"{p['conta']}   ({detalhe})")
    resumo = {"ok": True, "tipo": calculo["tipo"], "conta": calculo["expressao"], "passos": passos,
              "resultado": calculo["resultado"]}
    if "decimal" in calculo:
        resumo["decimal"] = calculo["decimal"]
    return resumo


def _chat_ollama(mensagens, ferramentas):
    """Uma chamada ao LLM. Devolve {"texto": ..., "chamadas": [{"nome", "argumentos"}]}."""
    import ollama  # importado aqui para os testes rodarem sem o Ollama
    resposta = ollama.chat(model=MODELO_LLM, messages=mensagens, tools=ferramentas,
                           options=OPCOES_LLM, keep_alive=MANTER_CARREGADO)
    chamadas = [{"nome": c.function.name, "argumentos": dict(c.function.arguments or {})}
                for c in (resposta.message.tool_calls or [])]
    return {"texto": resposta.message.content or "", "chamadas": chamadas}


def _mensagem_de_chamada(nome, argumentos):
    """Registra no histórico que o assistente chamou uma ferramenta (formato do Ollama)."""
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": nome, "arguments": argumentos}}]}


class Agente:
    """O tutor. Uso: agente.ler_conta(simbolos) e depois agente.responder("texto do visitante")."""

    def __init__(self, chat=_chat_ollama, limite=LIMITE_CONFIANCA):
        self.chat = chat            # pode ser trocado por um LLM falso nos testes
        self.limite = limite
        self.nova_conversa()

    def nova_conversa(self):
        self.historico = []         # mensagens trocadas com o LLM (sem o prompt de sistema)
        self.simbolos = []          # leitura atual da CNN
        self.duvidas = []           # posições ainda em dúvida
        self.conta = None           # conta confirmada (texto)
        self.ultimo_calculo = None
        self.exercicio = None       # exercício em aberto (com a resposta, que o LLM não vê)

    def aquecer(self):
        """Carrega o modelo no Ollama antes da primeira pergunta (a 1ª chamada é a mais lenta)."""
        try:
            # mesmo prompt e mesmas ferramentas: o Ollama guarda esse começo já processado
            self.chat([{"role": "system", "content": PROMPT_SISTEMA}, {"role": "user", "content": "Oi"}],
                      FERRAMENTAS_LLM)
            return True
        except Exception:
            return False

    # -- Entrada 1: uma conta lida pela CNN ----------------------------------
    def ler_conta(self, simbolos):
        """simbolos: lista de {"simbolo", "confianca", "top3"} (saída da segmentação)."""
        inicio = time.perf_counter()
        self.simbolos = [{"simbolo": s["simbolo"], "confianca": float(s.get("confianca", 1.0)),
                          "top3": list(s.get("top3") or [])} for s in simbolos]
        self.duvidas, self.conta, self.exercicio = [], None, None
        if not self.simbolos:
            return self._resposta("Não encontrei nenhum símbolo. Escreva a conta com caneta escura e mostre de novo.",
                                  "erro", inicio)
        # REGRA NO CÓDIGO: confiança baixa -> perguntar antes de calcular
        for i, s in enumerate(self.simbolos):
            opcoes = opcoes_da_duvida(s, self.limite)
            if opcoes:
                s["opcoes"] = opcoes
                self.duvidas.append(i)
        if self.duvidas:
            return self._perguntar_proxima(inicio)
        return self._conta_pronta(inicio)

    # -- Entrada 2: o visitante escreveu algo ---------------------------------
    def responder(self, mensagem):
        inicio = time.perf_counter()
        if self.duvidas:  # estamos esperando a resposta de uma pergunta
            i = self.duvidas[0]
            escolha = interpretar_resposta(mensagem, self.simbolos[i]["opcoes"])
            if escolha is None:
                return self._perguntar_proxima(inicio, prefixo="Desculpe, não entendi. ")
            self.duvidas.pop(0)
            if escolha == REMOVER:   # era sujeira: sai da conta (as posições seguintes andam uma casa)
                del self.simbolos[i]
                self.duvidas = [d - 1 if d > i else d for d in self.duvidas]
                if not self.simbolos:
                    return self._resposta("Então não sobrou nenhum símbolo. Mostre a conta de novo.", "erro", inicio)
            else:
                self.simbolos[i].update(simbolo=escolha, confianca=1.0)
            if self.duvidas:
                return self._perguntar_proxima(inicio, prefixo="Combinado! ")
            return self._conta_pronta(inicio)

        texto, veredito, exercicio = mensagem, None, None
        if self.exercicio:
            exercicio = self.exercicio["exercicio"]
            numeros = re.findall(r"-?\d+(?:/\d+)?", mensagem.replace("−", "-"))
            if numeros:  # REGRA NO CÓDIGO: quem confere a resposta é o SymPy, não o LLM
                certo = sp.Rational(re.findall(r"-?\d+(?:/\d+)?", self.exercicio["resposta"])[-1])
                veredito = {"exercicio": exercicio, "resposta_visitante": numeros[-1],
                            "acertou": sp.Rational(numeros[-1]) == certo}
                texto += (f"\n\n(Conferência feita pelo código: o visitante respondeu {numeros[-1]} para o "
                          f"exercício {exercicio} e {'ACERTOU' if veredito['acertou'] else 'ERROU'}. "
                          + ("Dê parabéns" if veredito["acertou"] else "Diga com carinho que não foi dessa vez")
                          + " e mostre os passos chamando calcular com o exercício.)")
            else:
                texto += f"\n\n(Exercício em aberto: {exercicio}. Não conte a resposta.)"
        return self._turno_llm(texto, inicio, conta_esperada=exercicio if veredito else None, veredito=veredito)

    # -- Etapas internas ------------------------------------------------------
    def _leitura_com_interrogacao(self):
        return "".join("?" if i in self.duvidas[:1] else s["simbolo"] for i, s in enumerate(self.simbolos))

    def _perguntar_proxima(self, inicio, prefixo=""):
        i = self.duvidas[0]
        s = self.simbolos[i]
        p = perguntar_ao_visitante(s["simbolo"], s["opcoes"], posicao=i + 1,
                                   leitura=self._leitura_com_interrogacao())
        return self._resposta(prefixo + p["pergunta"], "pergunta", inicio, pergunta=p, ferramentas=["perguntar_ao_visitante"])

    def _conta_pronta(self, inicio):
        ambiguidade = resolver_ambiguidade_x(self.simbolos)
        validacao = validar_expressao(ambiguidade["simbolos"])
        if not validacao["valida"]:
            texto = (f"Li **{ambiguidade['expressao']}**, mas acho que algo não está certo: "
                     + " ".join(validacao["problemas"])
                     + "\n\nConfira o papel e mostre de novo, ou escreva a conta aqui no chat.")
            # Fica no histórico: se o visitante corrigir a leitura, o LLM sabe de que conta ele fala
            self.historico += [{"role": "user", "content":
                                f"A câmera leu a conta {ambiguidade['expressao']}, mas ela não é válida: "
                                + " ".join(validacao["problemas"]) + " Se eu corrigir a leitura, "
                                "chame calcular com a conta corrigida, usando só os símbolos lidos."},
                               {"role": "assistant", "content": texto}]
            return self._resposta(texto, "erro", inicio, expressao=ambiguidade["expressao"])
        self.conta = ambiguidade["expressao"]
        pedido = f"O visitante mostrou a conta {self.conta}. Resolva com a ferramenta calcular e explique."
        mudancas = [d for d in ambiguidade["decisoes"] if d["mudou"]]
        for d in mudancas:
            pedido += f" (O {d['posicao']}º símbolo parecia '{d['lido']}', mas é '{d['decidido']}': {d['motivo']}.)"
        return self._turno_llm(pedido, inicio, conta_esperada=self.conta)

    def _executar(self, nome, argumentos, conta_esperada, calculos, eventos):
        """Executa a ferramenta pedida pelo LLM, com as travas do código."""
        expressao = str(argumentos.get("expressao") or self.conta or "")
        if nome == "calcular":
            if conta_esperada:  # trava: na conta nova, calcula exatamente o que a CNN leu
                expressao = conta_esperada
            calculo = calcular(resolver_ambiguidade_x(expressao)["simbolos"])
            if any(c.get("expressao") == calculo.get("expressao") for c in calculos):
                # conta já calculada neste turno: repetir só gasta tempo do LLM
                eventos["repetida"] = True
                return {"ok": True, "aviso": "Esta conta já foi calculada; o resultado está acima. "
                                             "Não chame calcular de novo: escreva a explicação."}
            calculos.append(calculo)
            return _resumo_para_llm(calculo)
        if nome == "gerar_exercicio":
            exercicio = gerar_exercicio(resolver_ambiguidade_x(expressao)["simbolos"])
            if not exercicio["ok"]:
                return exercicio
            eventos["exercicio"] = exercicio
            # o LLM NÃO recebe a resposta, para não "entregar" ao visitante
            return {"ok": True, "exercicio": exercicio["exercicio"],
                    "aviso": "Mostre o exercício e peça para o visitante tentar. Não diga a resposta."}
        if nome == "perguntar_ao_visitante":
            opcoes = [str(o) for o in (argumentos.get("opcoes") or [])] or ["?"]
            eventos["pergunta"] = perguntar_ao_visitante(str(argumentos.get("simbolo", "")), opcoes)
            return {"ok": True, "aviso": "Pergunta enviada. Espere a resposta do visitante."}
        return {"ok": False, "erro": f"Ferramenta desconhecida: {nome}"}

    def _turno_llm(self, pedido, inicio, conta_esperada=None, veredito=None):
        """Um turno do agente: o LLM decide, chama ferramentas, explica; o código confere."""
        turno = [{"role": "user", "content": pedido}]   # vai para o histórico se der certo
        rascunho = []                                    # tentativas reprovadas (não vão para o histórico)
        calculos, eventos, usadas, problemas_vistos, reprovadas = [], {}, [], [], []
        chamadas_llm, tentativas, forcado = 0, 0, False
        texto, aprovada, aviso = "", False, None
        ferramentas = FERRAMENTAS_LLM   # some se o LLM repetir uma conta já calculada

        for _ in range(MAX_RODADAS_FERRAMENTAS + MAX_TENTATIVAS + 1):
            mensagens = [{"role": "system", "content": PROMPT_SISTEMA}] + self._historico_recente() + turno + rascunho
            try:
                resposta = self.chat(mensagens, ferramentas)
            except Exception as erro:  # Ollama fechado, modelo não instalado, etc.
                aviso = f"LLM indisponível ({type(erro).__name__}); explicação automática."
                break
            chamadas_llm += 1

            if resposta["chamadas"]:  # o LLM decidiu usar ferramentas
                for c in resposta["chamadas"]:
                    resultado = self._executar(c["nome"], c["argumentos"], conta_esperada, calculos, eventos)
                    if eventos.pop("repetida", False):
                        # TRAVA: o LLM pediu a mesma conta de novo. Na próxima chamada ele fica
                        # sem ferramentas, então só pode escrever a explicação (antes, ele chegou
                        # a chamar calcular 6 vezes seguidas e o turno levou mais de 3 minutos).
                        ferramentas = []
                        usadas.append(c["nome"] + " (repetida)")
                    else:
                        usadas.append(c["nome"])
                    turno.append(_mensagem_de_chamada(c["nome"], c["argumentos"]))
                    turno.append({"role": "tool", "tool_name": c["nome"],
                                  "content": json.dumps(resultado, ensure_ascii=False)})
                if "pergunta" in eventos:
                    texto, aprovada = eventos["pergunta"]["pergunta"], True
                    break
                continue

            # TRAVA: conta nova e o LLM respondeu sem calcular -> o código chama calcular
            if conta_esperada and not calculos and not forcado:
                forcado = True
                resultado = self._executar("calcular", {"expressao": conta_esperada}, conta_esperada, calculos, eventos)
                usadas.append("calcular (forçado pelo código)")
                turno.append(_mensagem_de_chamada("calcular", {"expressao": conta_esperada}))
                turno.append({"role": "tool", "tool_name": "calcular",
                              "content": json.dumps(resultado, ensure_ascii=False)})
                continue

            # CHAIN OF VERIFICATION: conferir a explicação antes de mostrar
            texto = limpar_texto(resposta["texto"])
            tentativas += 1
            problemas = problemas_da_explicacao(texto, calculos[-1] if calculos else None,
                                                eventos.get("exercicio"), veredito,
                                                conhecidos=self._numeros_permitidos(pedido, calculos, eventos))
            if not problemas:
                aprovada = True
                break
            problemas_vistos.extend(problemas)
            reprovadas.append(texto)
            if tentativas >= MAX_TENTATIVAS:
                break
            rascunho += [{"role": "assistant", "content": texto},
                         {"role": "user", "content": "Revisão automática: " + " ".join(problemas)
                          + " Reescreva a explicação usando só os números da ferramenta calcular"
                            " (chame a ferramenta se precisar)."}]

        if not aprovada:  # plano B: explicação montada direto do SymPy
            if not calculos and conta_esperada:
                calculos.append(calcular(resolver_ambiguidade_x(conta_esperada)["simbolos"]))
            if calculos:
                texto = explicacao_automatica(calculos[-1])
                if veredito:
                    texto = ("Você acertou! Parabéns! 🎉" if veredito["acertou"] else
                             f"Quase! Não foi dessa vez: a resposta não é {veredito['resposta_visitante']}.") \
                        + "\n\n" + texto
            elif aviso:
                texto = "Desculpe, estou sem o meu cérebro de linguagem agora. Mostre uma conta que eu resolvo!"
            else:
                texto = "Desculpe, não consegui responder com certeza. Pode escrever a conta de novo?"
            aviso = aviso or "a explicação do LLM não passou na conferência; usei a explicação automática."

        self.historico += turno + [{"role": "assistant", "content": texto}]
        if calculos:
            self.ultimo_calculo = calculos[-1]
        if veredito:
            self.exercicio = None   # exercício conferido: fecha
        if "exercicio" in eventos:  # exercício novo: fica em aberto até o visitante responder
            self.exercicio = eventos["exercicio"]
        tipo = "pergunta" if "pergunta" in eventos else ("exercicio" if "exercicio" in eventos else "explicacao")
        return self._resposta(texto, tipo, inicio, expressao=conta_esperada, calculo=calculos[-1] if calculos else None,
                              ferramentas=usadas, chamadas_llm=chamadas_llm, pergunta=eventos.get("pergunta"),
                              veredito=veredito,
                              verificacao={"aprovada": aprovada, "tentativas": tentativas,
                                           "problemas": problemas_vistos, "reprovadas": reprovadas, "aviso": aviso})

    def _numeros_permitidos(self, pedido, calculos, eventos):
        """Tudo de onde a explicação pode tirar números: o que o visitante escreveu,
        os cálculos deste turno, o último cálculo da conversa e os exercícios."""
        partes = [pedido]
        for c in calculos + ([self.ultimo_calculo] if self.ultimo_calculo else []):
            partes.append(json.dumps(_resumo_para_llm(c), ensure_ascii=False))
        for e in (eventos.get("exercicio"), self.exercicio):
            if e:
                partes.append(str(e.get("exercicio", "")))
        return " ".join(partes)

    def _historico_recente(self):
        """Últimas mensagens, começando sempre numa mensagem do visitante."""
        recente = self.historico[-MAX_MENSAGENS:]
        while recente and recente[0]["role"] != "user":
            recente = recente[1:]
        return recente

    def _resposta(self, texto, tipo, inicio, **extras):
        return {"texto": texto, "tipo": tipo, "tempo": round(time.perf_counter() - inicio, 1),
                "expressao": extras.pop("expressao", None) or self.conta, **extras}


# ---------------------------------------------------------------------------
# 4. Teste pelo terminal
# ---------------------------------------------------------------------------

def simbolos_simulados(conta, duvidas=()):
    """Transforma "12+7*3" em símbolos como os da CNN, com confiança alta.
    duvidas: ["4=1"] -> o 4º símbolo fica com confiança 55% e 2º palpite "1"."""
    simbolos = [{"simbolo": EQUIVALENTES.get(c, c), "confianca": 0.99,
                 "top3": [(EQUIVALENTES.get(c, c), 0.99)]} for c in conta if not c.isspace()]
    for d in duvidas:
        posicao, alternativa = d.split("=")
        s = simbolos[int(posicao) - 1]
        s["confianca"] = 0.55
        s["top3"] = [(s["simbolo"], 0.55), (EQUIVALENTES.get(alternativa, alternativa), 0.40)]
    return simbolos


def _mostrar(resposta):
    print(f"\nTutor ({resposta['tempo']} s): {resposta['texto']}")
    detalhes = []
    if resposta.get("ferramentas"):
        detalhes.append("ferramentas: " + ", ".join(resposta["ferramentas"]))
    if resposta.get("chamadas_llm"):
        detalhes.append(f"chamadas ao LLM: {resposta['chamadas_llm']}")
    v = resposta.get("verificacao")
    if v:
        detalhes.append("conferência: " + ("✅ aprovada" if v["aprovada"] else "❌ reprovada")
                        + (f" na {v['tentativas']}ª tentativa" if v["tentativas"] else ""))
        for rascunho in v["reprovadas"]:
            detalhes.append(f"   reprovada: \"{rascunho[:90]}...\"")
        for p in v["problemas"]:
            detalhes.append(f"   motivo: {p}")
        if v["aviso"]:
            detalhes.append(f"   aviso: {v['aviso']}")
    for d in detalhes:
        print(f"   [{d}]")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Conversa com o tutor pelo terminal.")
    parser.add_argument("conta", nargs="?", help='conta digitada, por exemplo "12+7*3"')
    parser.add_argument("--duvida", action="append", default=[], metavar="POS=ALT",
                        help="simula confiança baixa: --duvida 4=1 (4º símbolo, 2º palpite 1)")
    parser.add_argument("--imagem", help="lê a conta de uma foto com a CNN")
    parser.add_argument("--sem-conversa", action="store_true", help="sai depois da primeira resposta")
    args = parser.parse_args()

    if args.imagem:
        from leitor_webcam import carregar_modelo
        from preprocessamento import carregar_imagem
        from segmentacao import ler_conta, montar_expressao
        simbolos = ler_conta(carregar_imagem(args.imagem), carregar_modelo())
        print("CNN leu:", montar_expressao(simbolos), " ".join(f"{s['simbolo']}:{s['confianca']:.0%}" for s in simbolos))
    elif args.conta:
        simbolos = simbolos_simulados(args.conta, args.duvida)
    else:
        parser.error("informe uma conta ou --imagem")

    agente = Agente()
    print(f"Carregando o {MODELO_LLM}...", end=" ", flush=True)
    t = time.perf_counter()
    print("pronto" if agente.aquecer() else "FALHOU (o Ollama está aberto?)", f"({time.perf_counter() - t:.1f} s)")

    _mostrar(agente.ler_conta(simbolos))
    if args.sem_conversa:
        return
    print("\n(Escreva sua mensagem. 'sair' para terminar.)")
    while True:
        try:
            mensagem = input("\nVocê: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if mensagem.lower() in {"sair", "q", "exit"}:
            break
        if mensagem:
            _mostrar(agente.responder(mensagem))


if __name__ == "__main__":
    main()
