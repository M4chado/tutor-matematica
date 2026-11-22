"""Arquitetura da CNN que reconhece os 17 símbolos (treinada do zero).

Ideia geral:
  - As camadas de CONVOLUÇÃO aprendem a enxergar padrões: primeiro traços
    e curvas simples, depois formas mais complexas (laços, cruzamentos).
  - O MAX POOLING reduz a imagem pela metade, guardando o mais importante.
  - A camada DENSA junta tudo e decide qual é o símbolo.
  - O SOFTMAX transforma a decisão em 17 probabilidades que somam 100%
    (é daí que vem a "confiança" de cada palpite).
"""
import keras
from keras import layers

from config import CLASSES, TAMANHO_IMAGEM


def bloco_convolucao(x, filtros):
    """Um bloco: convolução 3x3 -> batch normalization -> ReLU -> max pooling."""
    x = layers.Conv2D(filtros, (3, 3), padding="same", use_bias=False)(x)
    # Batch normalization: estabiliza e acelera o treino. Ela guarda médias
    # acumuladas para usar depois do treino; momentum=0.9 faz essas médias
    # acompanharem a rede mais depressa. Com o padrão (0.99) as médias ficavam
    # "atrasadas" e a rede acertava 93% treinando, mas só 29% sendo usada.
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.ReLU()(x)                 # deixa passar só os sinais positivos
    x = layers.MaxPooling2D((2, 2))(x)   # reduz a imagem pela metade
    return x


def criar_modelo(com_dropout=True):
    """Cria a CNN.

    com_dropout=False é usado na versão "que decora", para mostrar o overfitting.
    """
    entrada = keras.Input(shape=(TAMANHO_IMAGEM, TAMANHO_IMAGEM, 1))   # 45x45, 1 canal (cinza)

    x = bloco_convolucao(entrada, 32)    # 45x45 -> 22x22
    x = bloco_convolucao(x, 64)          # 22x22 -> 11x11
    x = bloco_convolucao(x, 128)         # 11x11 -> 5x5

    x = layers.Flatten()(x)              # transforma o "cubo" 5x5x128 numa lista de números
    x = layers.Dense(128, activation="relu")(x)
    if com_dropout:
        # Dropout desliga 50% dos neurônios ao acaso em cada passo do treino:
        # a rede não pode depender de um neurônio só, então precisa generalizar.
        x = layers.Dropout(0.5)(x)
    saida = layers.Dense(len(CLASSES), activation="softmax")(x)

    nome = "cnn_simbolos" if com_dropout else "cnn_decorando"
    return keras.Model(entrada, saida, name=nome)


if __name__ == "__main__":
    # Rodar "python src/modelo.py" mostra o resumo das camadas
    criar_modelo().summary()
