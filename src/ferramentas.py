"""Etapa 5: ferramentas confiáveis que o agente vai usar.

Aqui NÃO há LLM: só lógica exata. O LLM (Etapa 6) nunca faz contas sozinho;
ele chama estas funções e explica o resultado.

Ferramentas:
  - resolver_ambiguidade_x(simbolos): decide pelo contexto se x/× é multiplicação ou incógnita
  - validar_expressao(simbolos):      confere se a conta está bem escrita e explica o problema
  - calcular(expressao):              resolve com SymPy, passo a passo, sem eval()
  - gerar_exercicio(expressao):       cria uma conta parecida, com outros números

Segurança: NUNCA usamos eval() nem o parse_expr do SymPy (que usa eval por
dentro). Um analisador próprio, pequeno, lê os símbolos e monta a conta com
objetos do SymPy (Integer, Rational, Symbol).

Todas as funções devolvem dicionários simples (texto e números), prontos
para virar resposta de ferramenta do LLM.
"""
import random

import sympy as sp

X = sp.Symbol("x")
DIGITOS = set("0123456789")
OPERADORES = {"+", "-", "×", "÷"}
CRUZES = {"x", "×"}  # a CNN confunde os dois: o contexto decide
# Facilidades para digitar no teclado: * vira ×, / e : viram ÷, − vira -
EQUIVALENTES = {"*": "×", "/": "÷", ":": "÷", "−": "-", "X": "x"}
SIMBOLOS_VALIDOS = DIGITOS | OPERADORES | {"(", ")", "=", "x", "×"}


class ErroConta(Exception):
    """Problema na conta, com uma mensagem simples em português."""


# ---------------------------------------------------------------------------
# Entrada: aceita "12+7×3", ["1","2","+",...] ou a saída da segmentação
# ---------------------------------------------------------------------------

def _para_lista(simbolos):
    if isinstance(simbolos, str):
        lista = [c for c in simbolos if not c.isspace()]
    else:
        lista = [s["simbolo"] if isinstance(s, dict) else str(s) for s in simbolos]
    return [EQUIVALENTES.get(s, s) for s in lista]


def _compacto(lista):
    return "".join(lista)


# ---------------------------------------------------------------------------
# 1. Ambiguidade do x
# ---------------------------------------------------------------------------

def _e_fim_de_operando(s):
    return s in DIGITOS or s == ")"


def _e_inicio_de_operando(s):
    return s in DIGITOS or s == "("


def resolver_ambiguidade_x(simbolos):
    """Decide, para cada x ou ×, se é multiplicação (×) ou incógnita (x).

    Regras, nesta ordem:
      1. Entre dois números (ou parênteses), como em 3x4 ou (2+1)x5: é multiplicação.
      2. Se a conta tem "=", como em 2x+3=11: é a incógnita.
      3. Sem "=" não existe incógnita para descobrir: é multiplicação.
    """
    lista = _para_lista(simbolos)
    tem_igual = "=" in lista
    resultado, decisoes = list(lista), []
    for i, s in enumerate(lista):
        if s not in CRUZES:
            continue
        antes = lista[i - 1] if i > 0 else None
        depois = lista[i + 1] if i + 1 < len(lista) else None
        if antes and depois and _e_fim_de_operando(antes) and _e_inicio_de_operando(depois):
            novo, motivo = "×", f"está entre {antes} e {depois}, então é multiplicação"
        elif tem_igual:
            novo, motivo = "x", "a conta tem '=', então é a incógnita (o número que queremos descobrir)"
        else:
            novo, motivo = "×", "a conta não tem '=', então não há incógnita: é multiplicação"
        resultado[i] = novo
        decisoes.append({"posicao": i + 1, "lido": s, "decidido": novo, "motivo": motivo,
                         "mudou": novo != s})
    return {"simbolos": resultado, "expressao": _compacto(resultado), "decisoes": decisoes}


# ---------------------------------------------------------------------------
# 2. Validação
# ---------------------------------------------------------------------------

def _tokenizar(lista):
    """Junta dígitos seguidos num número só: ["1","2","+","3"] -> [12, "+", 3]."""
    tokens, numero = [], ""
    for s in lista:
        if s in DIGITOS:
            numero += s
            continue
        if numero:
            tokens.append(int(numero))
            numero = ""
        tokens.append(s)
    if numero:
        tokens.append(int(numero))
    return tokens


def _tipo(t):
    if isinstance(t, int):
        return "num"
    return {"x": "x", "(": "(", ")": ")", "=": "="}.get(t, "op")


# Multiplicação "escondida" permitida: 2x, 2(3+4), (1+2)(3+4), (1+2)x, x(2+1)
MULTIPLICACAO_IMPLICITA = {("num", "x"), ("num", "("), (")", "("), (")", "x"), ("x", "(")}


def validar_expressao(simbolos):
    """Confere se a conta está bem escrita. Devolve {"valida": bool, "problemas": [...]}.

    Atenção: x e × são lidos literalmente aqui (x = incógnita, × = vezes).
    Rode resolver_ambiguidade_x antes, quando os símbolos vierem da CNN.
    """
    lista = _para_lista(simbolos)
    problemas = []
    if not lista:
        return {"valida": False, "problemas": ["A conta está vazia: nenhum símbolo foi lido."]}

    desconhecidos = sorted({s for s in lista if s not in SIMBOLOS_VALIDOS})
    if desconhecidos:
        return {"valida": False, "problemas": [f"Símbolos que não conheço: {' '.join(desconhecidos)}"]}

    tokens = _tokenizar(lista)

    # Parênteses em par e na ordem certa
    abertos = 0
    for t in tokens:
        if t == "(":
            abertos += 1
        elif t == ")":
            abertos -= 1
            if abertos < 0:
                problemas.append("Tem um ')' fechando um parêntese que não foi aberto.")
                abertos = 0
    if abertos > 0:
        problemas.append(f"Falta fechar {abertos} parêntese(s): tem '(' sem o ')' correspondente.")

    # Sinal de igual
    qtd_igual = tokens.count("=")
    if qtd_igual > 1:
        problemas.append("Tem mais de um '='. A conta só pode ter um.")

    # Começo e fim de cada lado do "=" e pares de vizinhos
    for lado in _dividir_pelo_igual(tokens):
        if not lado:
            problemas.append("Um dos lados do '=' está vazio.")
            continue
        primeiro, ultimo = lado[0], lado[-1]
        if _tipo(primeiro) == "op" and primeiro != "-":
            problemas.append(f"A conta não pode começar com '{primeiro}'.")
        if _tipo(ultimo) == "op":
            problemas.append(f"A conta não pode terminar com '{ultimo}': falta um número depois dele.")
        if ultimo == "(":
            problemas.append("A conta não pode terminar com '('.")

    for a, b in zip(tokens, tokens[1:]):
        ta, tb = _tipo(a), _tipo(b)
        if ta == "op" and tb == "op":
            problemas.append(f"Dois operadores seguidos: '{a}{b}'. Falta um número entre eles.")
        elif ta == "(" and tb == ")":
            problemas.append("Tem parênteses vazios '()'.")
        elif ta == "(" and tb == "op" and b != "-":
            problemas.append(f"Depois de '(' não pode vir '{b}'.")
        elif ta == "op" and tb == ")":
            problemas.append(f"Antes de ')' não pode vir '{a}'.")
        elif ta in ("op", "(") and tb == "=" or ta == "=" and tb == ")":
            pass  # já coberto pelas regras de começo/fim de cada lado
        elif (ta, tb) in {(")", "num"), ("x", "num"), ("x", "x")}:
            problemas.append(f"'{a}' seguido de '{b}' sem operador entre eles: falta um sinal (+, -, ×, ÷)?")

    if "x" in tokens and qtd_igual == 0:
        problemas.append("A conta tem a incógnita x mas não tem '=': não há equação para resolver.")

    problemas = list(dict.fromkeys(problemas))  # remove repetidos, mantendo a ordem
    return {"valida": not problemas, "problemas": problemas, "expressao": _compacto(lista)}


def _dividir_pelo_igual(tokens):
    lados, atual = [], []
    for t in tokens:
        if t == "=":
            lados.append(atual)
            atual = []
        else:
            atual.append(t)
    lados.append(atual)
    return lados


# ---------------------------------------------------------------------------
# Analisador (sem eval): tokens -> árvore da conta
# ---------------------------------------------------------------------------
# Nós da árvore (tuplas simples):
#   ("num", Rational)          número
#   ("x",)                     incógnita
#   ("op", sinal, esq, dir, implicito)   operação (implicito=True em "2x" ou "2(3+4)")
#   ("neg", filho)             sinal de menos na frente: -3
#   ("par", filho)             parênteses escritos pelo usuário

class _Analisador:
    """Lê os tokens respeitando a prioridade das operações:
    expressao := termo (('+'|'-') termo)*
    termo     := unario (('×'|'÷'|implícita) unario)*
    unario    := '-' unario | primario
    primario  := número | x | '(' expressao ')'
    """

    def __init__(self, tokens):
        self.tokens, self.i = tokens, 0

    def espiar(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def consumir(self):
        t = self.espiar()
        self.i += 1
        return t

    def expressao(self):
        no = self.termo()
        while self.espiar() in ("+", "-"):
            sinal = self.consumir()
            no = ("op", sinal, no, self.termo(), False)
        return no

    def termo(self):
        no = self.unario()
        while True:
            t = self.espiar()
            if t in ("×", "÷"):
                self.consumir()
                no = ("op", t, no, self.unario(), False)
            elif t is not None and (self._ultimo_tipo(no), _tipo(t)) in MULTIPLICACAO_IMPLICITA:
                no = ("op", "×", no, self.unario(), True)
            else:
                return no

    def _ultimo_tipo(self, no):
        """Tipo do último token lido (para saber se cabe multiplicação implícita)."""
        anterior = self.tokens[self.i - 1]
        return _tipo(anterior)

    def unario(self):
        if self.espiar() == "-":
            self.consumir()
            return ("neg", self.unario())
        return self.primario()

    def primario(self):
        t = self.consumir()
        if isinstance(t, int):
            return ("num", sp.Integer(t))
        if t == "x":
            return ("x",)
        if t == "(":
            dentro = self.expressao()
            if self.consumir() != ")":
                raise ErroConta("Parênteses não fechados.")
            return ("par", dentro)
        raise ErroConta(f"Símbolo inesperado: {t}")


def _analisar_lado(tokens):
    analisador = _Analisador(tokens)
    arvore = analisador.expressao()
    if analisador.espiar() is not None:
        raise ErroConta(f"Não entendi a conta a partir de '{analisador.espiar()}'.")
    return arvore


# ---------------------------------------------------------------------------
# Mostrar a árvore como texto
# ---------------------------------------------------------------------------

def formatar_numero(valor):
    """33 -> "33";  7/2 -> "7/2";  -3 -> "-3"."""
    valor = sp.Rational(valor)
    return str(valor.p) if valor.q == 1 else f"{valor.p}/{valor.q}"


def decimal_br(valor):
    """7/2 -> "3,5" (vírgula, como no Brasil)."""
    texto = f"{float(valor):.4f}".rstrip("0").rstrip(".")
    return texto.replace(".", ",")


def _mostrar(no, filho_de_operacao=False, lado_direito=False):
    tipo = no[0]
    if tipo == "num":
        texto = formatar_numero(no[1])
        precisa_parenteses = filho_de_operacao and (no[1].q != 1 or (no[1] < 0 and lado_direito))
        return f"({texto})" if precisa_parenteses else texto
    if tipo == "x":
        return "x"
    if tipo == "neg":
        return "-" + _mostrar(no[1], True)
    if tipo == "par":
        return "(" + _mostrar(no[1]) + ")"
    _, sinal, esq, dir_, implicito = no
    e = _mostrar(esq, True, False)
    d = _mostrar(dir_, True, True)
    return f"{e}{d}" if implicito else f"{e} {sinal} {d}"


# ---------------------------------------------------------------------------
# Passo a passo das contas sem incógnita
# ---------------------------------------------------------------------------

def _simplificar(no):
    """Arruma o que não precisa de passo: -(3) vira -3 e (5) vira 5."""
    tipo = no[0]
    if tipo == "neg":
        filho = _simplificar(no[1])
        return ("num", -filho[1]) if filho[0] == "num" else ("neg", filho)
    if tipo == "par":
        filho = _simplificar(no[1])
        return filho if filho[0] in ("num", "x") else ("par", filho)
    if tipo == "op":
        _, sinal, esq, dir_, imp = no
        return ("op", sinal, _simplificar(esq), _simplificar(dir_), imp)
    return no


def _operacoes_prontas(no, profundidade=0, achadas=None):
    """Operações cujos dois lados já são números, em ordem da esquerda para a direita.
    Guarda também a profundidade de parênteses de cada uma."""
    if achadas is None:
        achadas = []
    if no[0] == "op":
        _, _, esq, dir_, _ = no
        _operacoes_prontas(esq, profundidade, achadas)
        if esq[0] == "num" and dir_[0] == "num":
            achadas.append((no, profundidade))
        _operacoes_prontas(dir_, profundidade, achadas)
    elif no[0] == "par":
        _operacoes_prontas(no[1], profundidade + 1, achadas)
    elif no[0] == "neg":
        _operacoes_prontas(no[1], profundidade, achadas)
    return achadas


def _tem_soma_fora_de_parenteses(no):
    if no[0] == "op":
        return no[1] in ("+", "-") or _tem_soma_fora_de_parenteses(no[2]) or _tem_soma_fora_de_parenteses(no[3])
    if no[0] == "neg":
        return _tem_soma_fora_de_parenteses(no[1])
    return False


def _substituir(no, alvo, novo):
    if no is alvo:
        return novo
    if no[0] == "op":
        _, sinal, esq, dir_, imp = no
        return ("op", sinal, _substituir(esq, alvo, novo), _substituir(dir_, alvo, novo), imp)
    if no[0] in ("par", "neg"):
        return (no[0], _substituir(no[1], alvo, novo))
    return no


def _fazer_operacao(sinal, a, b):
    if sinal == "+":
        return a + b
    if sinal == "-":
        return a - b
    if sinal == "×":
        return a * b
    if b == 0:
        raise ErroConta("Divisão por zero: não existe número que, vezes zero, dê outro número.")
    return sp.Rational(a, b)


def _passo_a_passo(arvore, sufixo=""):
    """Resolve uma operação por vez, na ordem ensinada na escola:
    1º parênteses (os mais de dentro), 2º × e ÷, 3º + e −, sempre da esquerda para a direita.
    Devolve (valor final, lista de passos)."""
    arvore = _simplificar(arvore)
    passos = []
    while arvore[0] != "num":
        prontas = _operacoes_prontas(arvore)
        if not prontas:
            raise ErroConta("Não consegui resolver a conta.")
        mais_funda = max(p for _, p in prontas)
        candidatas = [n for n, p in prontas if p == mais_funda]
        vezes_dividir = [n for n in candidatas if n[1] in ("×", "÷")]
        escolhida = (vezes_dividir or candidatas)[0]

        _, sinal, esq, dir_, _ = escolhida
        resultado = _fazer_operacao(sinal, esq[1], dir_[1])
        if escolhida is arvore:
            regra = "só falta esta operação"
        elif mais_funda > 0:
            regra = "primeiro o que está dentro dos parênteses"
        elif sinal in ("×", "÷") and _tem_soma_fora_de_parenteses(arvore):
            regra = "multiplicação e divisão vêm antes de soma e subtração"
        else:
            regra = "da esquerda para a direita"

        arvore = _simplificar(_substituir(arvore, escolhida, ("num", resultado)))
        operacao = f"{_mostrar(esq, True)} {sinal} {_mostrar(dir_, True, True)} = {formatar_numero(resultado)}"
        passos.append({"conta": _mostrar(arvore) + sufixo, "operacao": operacao, "regra": regra})
    return arvore[1], passos


# ---------------------------------------------------------------------------
# Equações do 1º grau
# ---------------------------------------------------------------------------

def _para_sympy(no):
    tipo = no[0]
    if tipo == "num":
        return no[1]
    if tipo == "x":
        return X
    if tipo in ("par",):
        return _para_sympy(no[1])
    if tipo == "neg":
        return -_para_sympy(no[1])
    _, sinal, esq, dir_, _ = no
    a, b = _para_sympy(esq), _para_sympy(dir_)
    if sinal == "+":
        return a + b
    if sinal == "-":
        return a - b
    if sinal == "×":
        return a * b
    if b == 0:
        raise ErroConta("Divisão por zero: não existe número que, vezes zero, dê outro número.")
    return a / b


def _linear(expr):
    """Escreve a·x + b na forma de coeficientes (a, b). Falha se não for do 1º grau."""
    expr = sp.expand(expr)
    if not expr.is_polynomial(X):
        raise ErroConta("O x aparece dividindo (no denominador): isso está fora do escopo do tutor.")
    polinomio = sp.Poly(expr, X)
    if polinomio.degree() > 1:
        raise ErroConta("Essa é uma equação de grau maior que 1 (tem x vezes x): fora do escopo do tutor.")
    return sp.Rational(polinomio.coeff_monomial(X)), sp.Rational(polinomio.coeff_monomial(1))


def _mostrar_linear(a, b):
    """(2, 3) -> "2x + 3";  (1, 0) -> "x";  (0, 5) -> "5";  (-1, -4) -> "-x - 4"."""
    partes = []
    if a != 0:
        coef = "" if a == 1 else "-" if a == -1 else (formatar_numero(a) if a.q == 1 else f"({formatar_numero(a)})")
        partes.append(f"{coef}x")
    if b != 0 or not partes:
        if partes:
            partes.append(f"{'+' if b > 0 else '-'} {formatar_numero(abs(b))}")
        else:
            partes.append(formatar_numero(b))
    return " ".join(partes)


def _resolver_equacao(esq, dir_, original):
    passos = [{"conta": original, "operacao": None, "regra": "equação original"}]
    a1, b1 = _linear(_para_sympy(esq))
    a2, b2 = _linear(_para_sympy(dir_))

    simplificada = f"{_mostrar_linear(a1, b1)} = {_mostrar_linear(a2, b2)}"
    if simplificada != original:
        passos.append({"conta": simplificada, "operacao": None,
                       "regra": "simplificamos cada lado (fazendo as contas e tirando os parênteses)"})

    # Termos com x para a esquerda, números para a direita
    a, c = a1 - a2, b2 - b1
    if a2 != 0 or b1 != 0:
        esquerda = f"{_mostrar_linear(a1, 0)} {'-' if a2 > 0 else '+'} {_mostrar_linear(abs(a2), 0)}" if a2 != 0 \
            else _mostrar_linear(a1, 0)
        direita = f"{formatar_numero(b2)} {'-' if b1 > 0 else '+'} {formatar_numero(abs(b1))}" if b1 != 0 \
            else formatar_numero(b2)
        passos.append({"conta": f"{esquerda} = {direita}", "operacao": None,
                       "regra": "passamos o que tem x para a esquerda e os números para a direita "
                                "(trocando o sinal de quem muda de lado)"})
        passos.append({"conta": f"{_mostrar_linear(a, 0)} = {formatar_numero(c)}" if a != 0
                       else f"0 = {formatar_numero(c)}", "operacao": None, "regra": "fazemos as contas de cada lado"})

    if a == 0:
        if c == 0:
            return {"tipo": "equacao", "solucao": "infinitas", "resultado": "qualquer número serve para x",
                    "valor": None, "passos": passos}
        return {"tipo": "equacao", "solucao": "nenhuma", "resultado": "nenhum número serve para x (sem solução)",
                "valor": None, "passos": passos}

    solucao = sp.Rational(c, a)
    if a != 1:
        divisor = formatar_numero(a) if a > 0 and a.q == 1 else f"({formatar_numero(a)})"
        passos.append({"conta": f"x = {formatar_numero(c)} ÷ {divisor}", "operacao": None,
                       "regra": f"dividimos os dois lados por {formatar_numero(a)} para deixar o x sozinho"})
        passos.append({"conta": f"x = {formatar_numero(solucao)}", "operacao": None, "regra": "fazemos a divisão"})

    # Conferência: troca o x pela solução nos dois lados
    lado_e = _para_sympy(esq).subs(X, solucao)
    lado_d = _para_sympy(dir_).subs(X, solucao)
    passos.append({"conta": f"conferindo: com x = {formatar_numero(solucao)}, "
                            f"o lado esquerdo dá {formatar_numero(lado_e)} e o direito dá {formatar_numero(lado_d)}",
                   "operacao": None, "regra": "conferência" + (" ✔" if lado_e == lado_d else " ✘")})
    return {"tipo": "equacao", "solucao": "unica", "resultado": f"x = {formatar_numero(solucao)}",
            "valor": formatar_numero(solucao), "passos": passos}


# ---------------------------------------------------------------------------
# 3. calcular
# ---------------------------------------------------------------------------

def calcular(expressao):
    """Resolve a conta e mostra os passos.

    Aceita expressões (12+7×3), equações do 1º grau (2x+3=11) e verificações
    (3+4=7: confere se é verdade). Lê x como incógnita e × como vezes;
    "*" e "/" também são aceitos.

    Devolve {"ok": True, "tipo", "expressao", "resultado", "valor", "passos"}
    ou {"ok": False, "erro": "..."}.
    """
    lista = _para_lista(expressao)
    validacao = validar_expressao(lista)
    if not validacao["valida"]:
        return {"ok": False, "erro": " ".join(validacao["problemas"]), "expressao": _compacto(lista)}

    try:
        lados = [_analisar_lado(lado) for lado in _dividir_pelo_igual(_tokenizar(lista))]
        original = " = ".join(_mostrar(lado) for lado in lados)

        if len(lados) == 1:  # expressão: 12 + 7 × 3
            valor, passos = _passo_a_passo(lados[0])
            resposta = {"tipo": "expressao", "resultado": formatar_numero(valor), "valor": formatar_numero(valor),
                        "passos": [{"conta": original, "operacao": None, "regra": "conta original"}] + passos}
            if valor.q != 1:
                resposta["decimal"] = decimal_br(valor)
            return {"ok": True, "expressao": original, **resposta}

        esq, dir_ = lados
        if "x" not in lista:  # verificação: 3 + 4 = 7
            ve, passos_e = _passo_a_passo(esq, sufixo=f" = {_mostrar(_simplificar(dir_))}")
            texto_e = formatar_numero(ve)
            vd, passos_d = _passo_a_passo(dir_, sufixo="")
            passos_d = [{**p, "conta": f"{texto_e} = {p['conta']}"} for p in passos_d]
            verdadeira = ve == vd
            final = f"{texto_e} {'=' if verdadeira else '≠'} {formatar_numero(vd)}"
            return {"ok": True, "tipo": "verificacao", "expressao": original, "verdadeira": verdadeira,
                    "resultado": "verdadeira" if verdadeira else "falsa",
                    "valor": "verdadeira" if verdadeira else "falsa",
                    "passos": [{"conta": original, "operacao": None, "regra": "conta original"}]
                    + passos_e + passos_d
                    + [{"conta": final, "operacao": None,
                        "regra": "os dois lados são iguais" if verdadeira else "os dois lados são diferentes"}]}

        return {"ok": True, "expressao": original, **_resolver_equacao(esq, dir_, original)}
    except ErroConta as erro:
        return {"ok": False, "erro": str(erro), "expressao": _compacto(lista)}


# ---------------------------------------------------------------------------
# 4. gerar_exercicio
# ---------------------------------------------------------------------------

def _novo_numero(original, gerador):
    """Número novo com a mesma quantidade de dígitos.
    De 1 dígito: de 2 a 9 (sem 0, que pode dar ÷ 0, e sem 1, que deixa a conta sem graça)."""
    digitos = len(str(original))
    if digitos == 1:
        return gerador.randint(2, 9)
    return gerador.randint(10 ** (digitos - 1), 10 ** digitos - 1)


def gerar_exercicio(expressao, semente=None, tentativas=1000):
    """Cria uma conta com a mesma "forma" e outros números.

    Garante uma conta boa para treinar: resultado inteiro (sem frações),
    não negativo se a original não era, e, nas equações, x inteiro.
    Numa verificação (3+4=7), o exercício é calcular o lado esquerdo.
    Devolve {"ok": True, "exercicio", "simbolos", "resposta"}.
    """
    lista = _para_lista(expressao)
    original = calcular(lista)
    if not original["ok"]:
        return {"ok": False, "erro": f"A conta original tem um problema: {original['erro']}"}

    if original["tipo"] == "verificacao":  # pratica só o lado esquerdo
        lista = lista[:lista.index("=")]
        original = calcular(lista)

    gerador = random.Random(semente)
    tokens = _tokenizar(lista)
    for _ in range(tentativas):
        novos = [_novo_numero(t, gerador) if isinstance(t, int) else t for t in tokens]
        simbolos = [c for t in novos for c in (str(t) if isinstance(t, int) else [t])]
        if simbolos == lista:
            continue
        r = calcular(simbolos)
        if not r["ok"] or r["tipo"] != original["tipo"]:
            continue
        if r["tipo"] == "expressao":
            valor = sp.Rational(r["valor"])
            if valor.q != 1 or abs(valor) > 1000:
                continue
            if sp.Rational(original["valor"]) >= 0 and valor < 0:
                continue
        elif r["tipo"] == "equacao":
            # x inteiro e pequeno (de 1 a 12, ou de -12 a 12 se a original tinha x negativo)
            if r["solucao"] != "unica" or sp.Rational(r["valor"]).q != 1:
                continue
            x = sp.Rational(r["valor"])
            positivo = original.get("solucao") != "unica" or sp.Rational(original["valor"]) > 0
            if abs(x) > 12 or (positivo and x < 1):
                continue
        return {"ok": True, "exercicio": r["expressao"], "simbolos": _compacto(simbolos),
                "resposta": r["resultado"]}
    return {"ok": False, "erro": "Não consegui criar um exercício parecido com números inteiros."}
