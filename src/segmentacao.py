"""Etapa 4: encontra cada símbolo de uma conta escrita em linha.

Caminho de uma foto/quadro da webcam:
  1. Igualar a iluminação (tira sombras) e separar tinta de papel.
  2. Encontrar os "pedaços" de tinta (componentes conectados).
  3. Descartar sujeira: pedaços minúsculos e o que encosta na borda.
     Exceção: pontinhos acima/abaixo de um traço deitado (os pontos do ÷).
  4. Juntar pedaços EMPILHADOS na mesma coluna: são partes do mesmo símbolo
     (os dois traços do =, os pontos do ÷, um 5 com o "chapéu" solto).
  5. Ordenar da esquerda para a direita.
  6. Recortar cada símbolo e aplicar a função ÚNICA preprocessar().

Uso típico:
    simbolos = ler_conta(imagem, modelo)
    print(montar_expressao(simbolos))    # ex.: "12+7×3"
"""
from pathlib import Path

import cv2
import numpy as np

from config import CLASSES, LIMITE_CONFIANCA
from preprocessamento import normalizar_iluminacao, preprocessar

AREA_MINIMA = 4           # pedaços com menos pixels que isso são sempre poeira
FRACAO_PEQUENO = 0.25     # menor que 25% do tamanho típico de símbolo = "pequeno"
SOBREPOSICAO_X = 0.5      # para juntar, as colunas precisam se sobrepor em 50% (do mais estreito)
SOBREPOSICAO_Y_MAX = 0.5  # ...e quase não se sobrepor na vertical (estão um em cima do outro)
MARGEM_RECORTE = 3


# ---------------------------------------------------------------------------
# Detecção da folha: só vale a tinta que está CERCADA DE PAPEL
# ---------------------------------------------------------------------------
LADO_DETECCAO = 640        # a folha é procurada numa versão pequena da foto (mais rápido)
SATURACAO_PAPEL = 60       # papel é branco/cinza: pouca cor (0 a 255)
BRILHO_MIN_PAPEL = 100     # e é claro: abaixo disso (0 a 255) não é papel
AREA_MIN_FOLHA = 0.02      # uma região de papel ocupa pelo menos 2% da foto
RETANGULAR_MIN = 0.70      # e tem formato de retângulo (área / retângulo que a envolve)
SATURACAO_COR = 70         # tinta com mais cor que isso e matiz quente é estampa/pele, não caneta


def mascara_da_folha(imagem_bgr, forma):
    """Onde está o papel? Devolve um mapa do tamanho `forma` (altura, largura) em
    que cada região de papel tem um número (1, 2, ...) e o resto é 0, ou None se
    nenhuma folha for encontrada.

    1. Papel = claro e sem cor (pouca saturação).
    2. As bordas fortes (contorno da folha, dedos, prateleiras) cortam as regiões,
       separando a folha do que está em volta.
    3. Só ficam regiões grandes e com formato de retângulo (a folha).
    4. Os "buracos" (a tinta escrita) são preenchidos: fazem parte da folha.
    """
    if imagem_bgr.ndim == 2:
        imagem_bgr = cv2.cvtColor(imagem_bgr, cv2.COLOR_GRAY2BGR)
    escala = LADO_DETECCAO / max(imagem_bgr.shape[:2])
    pequena = cv2.resize(imagem_bgr, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(pequena, cv2.COLOR_BGR2HSV)
    brilho = cv2.GaussianBlur(hsv[..., 2], (5, 5), 0)
    limiar, _ = cv2.threshold(brilho, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    claro = brilho >= max(limiar, BRILHO_MIN_PAPEL)  # numa imagem toda escura, nada é papel
    bordas = cv2.dilate(cv2.Canny(brilho, 20, 60), np.ones((3, 3), np.uint8))
    papel = (claro & (hsv[..., 1] < SATURACAO_PAPEL) & (bordas == 0)).astype(np.uint8)
    papel = cv2.morphologyEx(papel, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # corta pontes finas

    folha = np.zeros(papel.shape, np.uint8)
    n, rotulos, estatisticas, _ = cv2.connectedComponentsWithStats(papel, connectivity=8)
    for i in range(1, n):
        if estatisticas[i, cv2.CC_STAT_AREA] < AREA_MIN_FOLHA * papel.size:
            continue
        contornos, _ = cv2.findContours((rotulos == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        contorno = max(contornos, key=cv2.contourArea)
        _, (w, h), _ = cv2.minAreaRect(contorno)
        if cv2.contourArea(contorno) / max(w * h, 1) >= RETANGULAR_MIN:
            numero = int(folha.max()) + 1
            cv2.drawContours(folha, [contorno], -1, numero, thickness=cv2.FILLED)  # preenche os buracos
    if not folha.any():
        return None
    return cv2.resize(folha, (forma[1], forma[0]), interpolation=cv2.INTER_NEAREST)


def _cor_de_caneta(pixels_bgr):
    """Tinta de caneta é escura, preta ou azul. Vermelho, laranja, amarelo e cor
    de pele (logotipos, dedos) são descartados."""
    matiz, saturacao, _ = np.median(cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV), axis=0)[0]
    quente = matiz < 35 or matiz > 155       # escala do OpenCV: 0 a 180
    return not (saturacao > SATURACAO_COR and quente)


def binarizar_foto(limpa):
    """True onde tem tinta. Otsu escolhe o limite; o teto de 200 evita que
    um papel levemente cinza vire "tinta" quando quase não há traço."""
    limiar, _ = cv2.threshold(limpa, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # "<=" e não "<": numa imagem com só dois tons, o Otsu põe o limite
    # exatamente no tom da tinta, e "<" deixaria a tinta de fora
    return limpa <= min(limiar, 200)


class _Grupo:
    """Um símbolo em formação: um ou mais pedaços de tinta e o retângulo que os envolve."""

    def __init__(self, rotulo, x, y, w, h):
        self.rotulos = [rotulo]
        self.x0, self.y0, self.x1, self.y1 = x, y, x + w, y + h

    @property
    def largura(self):
        return self.x1 - self.x0

    @property
    def altura(self):
        return self.y1 - self.y0

    @property
    def tamanho(self):
        return max(self.largura, self.altura)

    def absorver(self, outro):
        self.rotulos += outro.rotulos
        self.x0, self.y0 = min(self.x0, outro.x0), min(self.y0, outro.y0)
        self.x1, self.y1 = max(self.x1, outro.x1), max(self.y1, outro.y1)


def _sobreposicao(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _empilhados(a, b):
    """Dois pedaços são do mesmo símbolo se estão na mesma coluna, um em cima do outro."""
    sx = _sobreposicao(a.x0, a.x1, b.x0, b.x1)
    sy = _sobreposicao(a.y0, a.y1, b.y0, b.y1)
    mesma_coluna = sx >= SOBREPOSICAO_X * min(a.largura, b.largura)
    um_sobre_outro = sy <= SOBREPOSICAO_Y_MAX * min(a.altura, b.altura)
    return mesma_coluna and um_sobre_outro


def _e_ponto_do_dividir(pequeno, grupos, referencia):
    """Um pontinho só é mantido se estiver acima/abaixo de um traço deitado (como no ÷)."""
    cx = (pequeno.x0 + pequeno.x1) / 2
    cy = (pequeno.y0 + pequeno.y1) / 2
    for g in grupos:
        deitado = g.largura >= 2 * max(g.altura, 1)
        centro_y = (g.y0 + g.y1) / 2
        if deitado and g.x0 <= cx <= g.x1 and abs(cy - centro_y) <= 0.8 * max(g.largura, referencia):
            return True
    return False


def segmentar(imagem, so_na_folha=False):
    """Encontra os símbolos de uma conta. Devolve uma lista (da esquerda para a
    direita) de dicionários com:
      - "caixa": (x, y, largura, altura) na imagem ORIGINAL
      - "recorte": o símbolo recortado (cinza, fundo branco)
      - "entrada": o recorte já pela função única, pronto para a CNN (45, 45, 1)

    so_na_folha=True (fotos e webcam): só aceita tinta cercada de papel e com cor
    de caneta. Assim mão, roupa, prateleira e o que mais estiver em volta da
    folha são ignorados. Se nenhuma folha for encontrada, lê a imagem toda.
    """
    imagem = np.asarray(imagem)
    largura_original = imagem.shape[1]
    limpa = normalizar_iluminacao(imagem)           # pode reduzir fotos grandes
    escala = largura_original / limpa.shape[1]      # para voltar às coordenadas originais
    altura_img, largura_img = limpa.shape

    folha = mascara_da_folha(imagem, limpa.shape) if so_na_folha else None
    regiao_do_pedaco = {}  # em qual região de papel está cada pedaço de tinta
    if folha is not None:
        # O limite tinta/papel é calculado só DENTRO da folha (o fundo não atrapalha)
        no_papel = folha > 0
        limiar, _ = cv2.threshold(limpa[no_papel].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tinta = ((limpa <= min(limiar, 200)) & no_papel).astype(np.uint8)
        cores = cv2.resize(imagem if imagem.ndim == 3 else cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR),
                           (largura_img, altura_img), interpolation=cv2.INTER_AREA)
        margem = np.ones((7, 7), np.uint8)  # o papel precisa cercar a tinta com folga de 3 pixels
    else:
        tinta = binarizar_foto(limpa).astype(np.uint8)

    n, rotulos, estatisticas, _ = cv2.connectedComponentsWithStats(tinta, connectivity=8)

    # 2 e 3a. Pedaços de tinta, sem os que tocam a borda e sem poeira óbvia
    pedacos = []
    for i in range(1, n):  # o rótulo 0 é o papel
        x, y, w, h, area = estatisticas[i]
        toca_borda = x == 0 or y == 0 or x + w >= largura_img or y + h >= altura_img
        if toca_borda or area < AREA_MINIMA:
            continue
        if folha is not None:
            y0, y1, x0, x1 = max(y - 3, 0), y + h + 3, max(x - 3, 0), x + w + 3
            este = rotulos[y0:y1, x0:x1] == i
            em_volta = folha[y0:y1, x0:x1][cv2.dilate(este.astype(np.uint8), margem) > 0]
            cercado = em_volta.min() > 0 and em_volta.min() == em_volta.max()  # papel de uma região só
            if not cercado or not _cor_de_caneta(cores[y0:y1, x0:x1][este]):
                continue
            regiao_do_pedaco[i] = int(em_volta[0])
        pedacos.append(_Grupo(i, x, y, w, h))

    # A conta está numa folha só: fica a região de papel com mais tinta escrita
    # (outra região clara, como uma camisa branca, pode ter costuras e dobras)
    if regiao_do_pedaco and pedacos:
        tinta_por_regiao = {}
        for p in pedacos:
            r = regiao_do_pedaco[p.rotulos[0]]
            tinta_por_regiao[r] = tinta_por_regiao.get(r, 0) + int(estatisticas[p.rotulos[0], cv2.CC_STAT_AREA])
        escolhida = max(tinta_por_regiao, key=tinta_por_regiao.get)
        pedacos = [p for p in pedacos if regiao_do_pedaco[p.rotulos[0]] == escolhida]
    if not pedacos:
        return []

    # 3b. Tamanho típico de um símbolo (mediana dos pedaços razoavelmente grandes)
    maior = max(p.tamanho for p in pedacos)
    referencia = float(np.median([p.tamanho for p in pedacos if p.tamanho >= 0.3 * maior]))
    grandes = [p for p in pedacos if p.tamanho >= FRACAO_PEQUENO * referencia]
    pequenos = [p for p in pedacos if p.tamanho < FRACAO_PEQUENO * referencia]
    grupos = grandes + [p for p in pequenos if _e_ponto_do_dividir(p, grandes, referencia)]

    # 4. Junta pedaços empilhados, repetindo até não haver mais o que juntar
    juntou = True
    while juntou:
        juntou = False
        for i in range(len(grupos)):
            for j in range(i + 1, len(grupos)):
                if _empilhados(grupos[i], grupos[j]):
                    grupos[i].absorver(grupos.pop(j))
                    juntou = True
                    break
            if juntou:
                break

    # 5. Esquerda para a direita
    grupos.sort(key=lambda g: g.x0)

    # 6. Recorta cada símbolo (só os pixels dele) e aplica a função única
    simbolos = []
    for g in grupos:
        y0, y1 = max(g.y0 - MARGEM_RECORTE, 0), min(g.y1 + MARGEM_RECORTE, altura_img)
        x0, x1 = max(g.x0 - MARGEM_RECORTE, 0), min(g.x1 + MARGEM_RECORTE, largura_img)
        so_este = np.isin(rotulos[y0:y1, x0:x1], g.rotulos)  # ignora pedaços de vizinhos
        recorte = np.where(so_este, limpa[y0:y1, x0:x1], 255).astype(np.uint8)
        caixa = tuple(int(round(v * escala)) for v in (g.x0, g.y0, g.largura, g.altura))
        simbolos.append({"caixa": caixa, "recorte": recorte, "entrada": preprocessar(recorte)})
    return simbolos


def classificar(simbolos, modelo):
    """Passa todos os símbolos pela CNN de uma vez e acrescenta a cada um:
    "simbolo" (palpite), "confianca" (0 a 1) e "top3" [(símbolo, prob), ...]."""
    if not simbolos:
        return simbolos
    entradas = np.stack([s["entrada"] for s in simbolos])
    probabilidades = modelo.predict(entradas, verbose=0)
    for s, p in zip(simbolos, probabilidades):
        ordem = np.argsort(p)[::-1]
        s["simbolo"] = CLASSES[ordem[0]]
        s["confianca"] = float(p[ordem[0]])
        s["top3"] = [(CLASSES[i], float(p[i])) for i in ordem[:3]]
    return simbolos


def ler_conta(imagem, modelo, so_na_folha=False):
    """Segmenta e classifica: o caminho completo imagem -> lista de símbolos."""
    return classificar(segmentar(imagem, so_na_folha), modelo)


def montar_expressao(simbolos):
    """Junta os palpites na ordem: [1, 2, +, 7] -> "12+7"."""
    return "".join(s["simbolo"] for s in simbolos)


# ---------------------------------------------------------------------------
# Desenho das caixas (usado pela webcam e, depois, pela interface)
# ---------------------------------------------------------------------------

def _fonte(tamanho):
    """Fonte com suporte a × e ÷ (o cv2.putText só escreve letras simples).
    Usa a fonte DejaVu que já vem com o Matplotlib, então funciona offline."""
    from PIL import ImageFont
    import matplotlib

    caminho = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
    return ImageFont.truetype(str(caminho), tamanho)


def desenhar_leitura(imagem_bgr, simbolos, deslocamento=(0, 0)):
    """Desenha uma caixa em cada símbolo com o palpite e a confiança.
    Verde = confiança boa; laranja = abaixo de LIMITE_CONFIANCA (duvidoso)."""
    from PIL import Image, ImageDraw

    dx, dy = deslocamento
    imagem = Image.fromarray(cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB))
    desenho = ImageDraw.Draw(imagem)
    tamanho_fonte = max(14, imagem.height // 28)
    fonte = _fonte(tamanho_fonte)
    for s in simbolos:
        x, y, w, h = s["caixa"]
        x, y = x + dx, y + dy
        cor = (27, 175, 122) if s.get("confianca", 1) >= LIMITE_CONFIANCA else (235, 104, 52)
        desenho.rectangle([x, y, x + w, y + h], outline=cor, width=3)
        if "simbolo" in s:
            texto = f"{s['simbolo']} {100 * s['confianca']:.0f}%"
            topo = max(0, y - tamanho_fonte - 8)
            caixa_texto = desenho.textbbox((x, topo), texto, font=fonte)
            desenho.rectangle([caixa_texto[0] - 3, caixa_texto[1] - 2, caixa_texto[2] + 3, caixa_texto[3] + 2],
                              fill=cor)
            desenho.text((x, topo), texto, font=fonte, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(imagem), cv2.COLOR_RGB2BGR)
