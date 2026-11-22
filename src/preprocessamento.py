"""Função ÚNICA de pré-processamento: imagem qualquer -> entrada da CNN.

Ela é usada tanto para as imagens do dataset quanto para os recortes da webcam.
Se cada lado tivesse o seu próprio tratamento, a rede veria imagens "diferentes"
na feira e erraria mesmo estando bem treinada.

Passos:
  1. Tons de cinza.
  2. Polaridade igual à do dataset: traço ESCURO sobre fundo CLARO.
  3. Separar tinta e fundo (preto e branco puro).
  4. Recortar só a área do símbolo e centralizá-lo num quadrado.
  5. Reduzir para 45x45 sem perder traços finos.
  6. Afinar o traço até 1 pixel (o dataset tem traço fino; a caneta da
     webcam é grossa, então afinamos os dois para ficarem iguais).
  7. Valores entre 0 e 1 (fundo = 1, traço = 0), formato (45, 45, 1).
"""
import cv2
import numpy as np

TAMANHO = 45  # mesmo valor de config.TAMANHO_IMAGEM


def preprocessar(imagem):
    """Transforma uma imagem (numpy, cinza ou colorida, qualquer tamanho)
    na entrada da CNN: array float32 de formato (45, 45, 1), valores 0 a 1."""
    cinza = _para_cinza(imagem)
    tinta = _separar_tinta(cinza)          # True onde tem traço

    # Imagem sem nenhum traço: devolve um quadrado todo branco
    if not tinta.any():
        return np.ones((TAMANHO, TAMANHO, 1), dtype=np.float32)

    quadrado = _recortar_e_centralizar(tinta)
    pequena = _reduzir(quadrado)
    fina = _afinar(pequena)

    # Fundo claro (1.0) e traço escuro (0.0), como no dataset
    saida = np.where(fina, 0.0, 1.0).astype(np.float32)
    return saida[:, :, np.newaxis]


# ---------------------------------------------------------------------------
# Funções auxiliares (o "_" no nome indica que são de uso interno)
# ---------------------------------------------------------------------------

def _para_cinza(imagem):
    """Converte para tons de cinza (0 = preto, 255 = branco), em uint8."""
    imagem = np.asarray(imagem)
    if imagem.ndim == 3 and imagem.shape[2] == 4:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGRA2GRAY)
    elif imagem.ndim == 3 and imagem.shape[2] == 3:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    elif imagem.ndim == 3:
        imagem = imagem[:, :, 0]
    # Se vier em 0..1 (float), passa para 0..255
    if imagem.dtype != np.uint8:
        imagem = np.asarray(imagem, dtype=np.float64)
        if imagem.max() <= 1.0:
            imagem = imagem * 255
        imagem = np.clip(imagem, 0, 255).astype(np.uint8)
    return imagem


def _separar_tinta(cinza):
    """Devolve uma máscara True/False: True onde tem traço.

    Também acerta a polaridade: se o fundo for escuro (traço claro),
    a imagem é invertida. O fundo é descoberto olhando a borda da imagem,
    que quase sempre é papel.
    """
    borda = np.concatenate([cinza[0, :], cinza[-1, :], cinza[:, 0], cinza[:, -1]])
    if np.median(borda) < 128:
        cinza = 255 - cinza  # fundo escuro -> inverte para fundo claro

    # Pouca variação = imagem lisa, sem traço nenhum
    if int(cinza.max()) - int(cinza.min()) < 30:
        return np.zeros(cinza.shape, dtype=bool)

    # Otsu escolhe sozinho o melhor limite entre "tinta" e "papel"
    _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binaria > 0


def _recortar_e_centralizar(tinta):
    """Recorta o retângulo que contém o símbolo e o coloca no centro de um
    quadrado, sem esticar (um "-" continua achatado, um "1" continua alto)."""
    linhas = np.where(tinta.any(axis=1))[0]
    colunas = np.where(tinta.any(axis=0))[0]
    recorte = tinta[linhas[0]:linhas[-1] + 1, colunas[0]:colunas[-1] + 1]

    altura, largura = recorte.shape
    lado = max(altura, largura)
    quadrado = np.zeros((lado, lado), dtype=bool)
    topo = (lado - altura) // 2
    esquerda = (lado - largura) // 2
    quadrado[topo:topo + altura, esquerda:esquerda + largura] = recorte
    return quadrado


def _reduzir(quadrado):
    """Reduz o quadrado para 45x45 sem apagar traços finos.

    Ao encolher muito uma imagem grande, um traço fino viraria um cinza
    fraquinho e sumiria. Por isso o traço é engrossado antes, na mesma
    proporção da redução.
    """
    lado = quadrado.shape[0]
    imagem = quadrado.astype(np.uint8) * 255
    fator = lado / TAMANHO
    if fator > 1:
        k = int(np.ceil(fator))
        imagem = cv2.dilate(imagem, np.ones((k, k), np.uint8))
    pequena = cv2.resize(imagem, (TAMANHO, TAMANHO), interpolation=cv2.INTER_AREA)
    return pequena >= 128


def _afinar(tinta):
    """Afina o traço até ficar com 1 pixel de largura, sem quebrá-lo.

    Usa o algoritmo clássico de Zhang-Suen: em cada rodada, apaga os pixels
    da "casca" do traço que podem sair sem desconectar nada, até não sobrar
    nada para apagar.
    """
    img = np.pad(tinta.astype(np.uint8), 1)  # moldura de zeros para olhar os vizinhos
    while True:
        mudou = False
        for passo in (0, 1):
            # Os 8 vizinhos de cada pixel, em sentido horário começando em cima
            p2 = img[:-2, 1:-1]; p3 = img[:-2, 2:]; p4 = img[1:-1, 2:]; p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]; p7 = img[2:, :-2]; p8 = img[1:-1, :-2]; p9 = img[:-2, :-2]
            vizinhos = [p2, p3, p4, p5, p6, p7, p8, p9]

            quantos = sum(v.astype(np.int16) for v in vizinhos)  # vizinhos com tinta
            # Quantas vezes passa de "sem tinta" para "com tinta" dando a volta
            transicoes = sum(
                ((vizinhos[i] == 0) & (vizinhos[(i + 1) % 8] == 1)).astype(np.int16)
                for i in range(8)
            )
            if passo == 0:
                extra = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                extra = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

            centro = img[1:-1, 1:-1]
            apagar = (centro == 1) & (quantos >= 2) & (quantos <= 6) & (transicoes == 1) & extra
            if apagar.any():
                centro[apagar] = 0
                mudou = True
        if not mudou:
            break
    return img[1:-1, 1:-1].astype(bool)


def carregar_imagem(caminho):
    """Lê uma imagem do disco em tons de cinza.

    Usa np.fromfile + cv2.imdecode porque o cv2.imread falha no Windows
    quando o caminho tem acentos (ex.: "Documentos/Matemática").
    """
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        raise ValueError(f"Não foi possível ler a imagem: {caminho}")
    return imagem
