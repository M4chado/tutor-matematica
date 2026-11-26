# Textos do banner: CalcuLetra

Para copiar e colar no modelo do 5º Workshop de Trabalhos de IA (UniRV/FASOFT).
Os mesmos textos estão no topo de `gerar_banner.py`. Se mudar aqui, mude lá também.

## Título (2 linhas)

CalcuLetra: Tutor de Matemática que Lê sua Letra
com Rede Neural e Agente de IA Local

## Autores

Eduardo Machado Martins (eduardo.martins@academico.unirv.edu.br)
Augusto Wobeto Carvalho (augusto.carvalho@academico.unirv.edu.br)

## Introdução

Aprender matemática envolve escrever contas à mão, mas a maioria das ferramentas digitais exige que o aluno digite a expressão. Além disso, os modelos de linguagem (LLMs) explicam bem, mas podem errar cálculos, pois geram texto prevendo palavras, e não calculando.

Este trabalho apresenta o CalcuLetra, um tutor que lê contas escritas à mão por uma webcam e as resolve e explica passo a passo, em português. O sistema une duas áreas da Inteligência Artificial: uma rede neural convolucional (CNN), treinada do zero, reconhece os símbolos; e um agente com LLM local decide o que fazer, usa ferramentas de cálculo exato e confere a própria resposta. O objetivo é mostrar que a rede realmente aprende, e não apenas decora, e que um agente bem projetado evita respostas erradas, funcionando inteiramente sem internet.

## Metodologia

Foram usadas 34.722 imagens únicas do dataset Handwritten Math Symbols (Kaggle), após a remoção de 77% de cópias repetidas, divididas em treino (70%), validação (15%) e teste (15%), com 17 classes: dígitos, + − × ÷ = ( ) e x. A CNN tem três blocos de convolução (32, 64 e 128 filtros) com batch normalization e foi treinada na CPU com data augmentation, dropout e early stopping. Uma versão sem essas proteções, treinada com 5% dos dados, serviu de comparação.

Na webcam, o OpenCV encontra a folha e recorta cada símbolo. O agente usa o LLM qwen3:4b-instruct (Ollama) com tool calling: pergunta quando a confiança fica abaixo de 80%, calcula com o SymPy e confere se cada número da explicação está certo e veio da ferramenta (Chain of Verification).

Figura: `figura_metodologia.png`

## Resultados

No conjunto de teste, usado uma única vez, a CNN acertou 94,5% das imagens, valor próximo ao da validação (94,9%), o que indica generalização. O modelo que decora chegou a 99,9% no treino, mas só 90,3% na validação. A maior confusão foi entre a incógnita x e o sinal ×, resolvida pelo contexto da conta. Em testes reais pela webcam, 77% dos símbolos foram lidos corretamente e, quando havia dúvida, o tutor perguntou ao usuário. Os testes também revelaram uma alucinação: após uma leitura errada, o LLM inventou uma conta que não estava na folha, com cálculos corretos. A conferência passou a rejeitar números que não vieram da ferramenta de cálculo.

Figura: `figura_resultados.png`

## Referências (ABNT, em ordem alfabética)

DHULIAWALA, S. et al. Chain-of-Verification reduces hallucination in large language models. arXiv preprint arXiv:2309.11495, 2023.

HANDWRITTEN math symbols dataset. Kaggle, [s.d.]. Disponível em: https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols. Acesso em: 23 nov. 2026.

LECUN, Y. et al. Gradient-based learning applied to document recognition. Proceedings of the IEEE, v. 86, n. 11, p. 2278-2324, 1998.

MEURER, A. et al. SymPy: symbolic computing in Python. PeerJ Computer Science, v. 3, e103, 2017.
