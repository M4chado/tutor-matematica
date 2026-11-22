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


def segmentar(imagem):
    """Encontra os símbolos de uma conta. Devolve uma lista (da esquerda para a
    direita) de dicionários com:
      - "caixa": (x, y, largura, altura) na imagem ORIGINAL
      - "recorte": o símbolo recortado (cinza, fundo branco)
      - "entrada": o recorte já pela função única, pronto para a CNN (45, 45, 1)
    """
    largura_original = np.asarray(imagem).shape[1]
    limpa = normalizar_iluminacao(imagem)           # pode reduzir fotos grandes
    escala = largura_original / limpa.shape[1]      # para voltar às coordenadas originais
    tinta = binarizar_foto(limpa).astype(np.uint8)

    n, rotulos, estatisticas, _ = cv2.connectedComponentsWithStats(tinta, connectivity=8)
    altura_img, largura_img = tinta.shape

    # 2 e 3a. Pedaços de tinta, sem os que tocam a borda e sem poeira óbvia
    pedacos = []
    for i in range(1, n):  # o rótulo 0 é o papel
        x, y, w, h, area = estatisticas[i]
        toca_borda = x == 0 or y == 0 or x + w >= largura_img or y + h >= altura_img
        if not toca_borda and area >= AREA_MINIMA:
            pedacos.append(_Grupo(i, x, y, w, h))
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


def ler_conta(imagem, modelo):
    """Segmenta e classifica: o caminho completo imagem -> lista de símbolos."""
    return classificar(segmentar(imagem), modelo)


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
