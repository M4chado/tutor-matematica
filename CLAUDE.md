# CLAUDE.md — CalcuLetra (Tutor de Matemática que Lê sua Letra)

Este arquivo orienta o Claude Code no desenvolvimento deste projeto. Leia-o inteiro antes de qualquer ação e siga-o em todas as sessões.

## 1. O projeto

Projeto individual de graduação para uma feira de engenharia de software. Ele combina duas matérias:

- **Redes Neurais:** uma CNN treinada do zero reconhece símbolos matemáticos escritos à mão.
- **LLMs e Agentes Inteligentes:** um agente com LLM local resolve, explica e confere a conta.

Fluxo completo: webcam fotografa uma conta escrita no papel → OpenCV recorta cada símbolo → a CNN classifica cada símbolo com uma confiança → a expressão vai para o agente → o agente decide (perguntar, calcular, explicar) → a interface mostra a resposta em texto e, opcionalmente, em voz.

**Critério de avaliação mais importante:** o professor vai verificar se o modelo **aprende de verdade ou apenas decora**. Todo o projeto deve produzir evidências disso: divisão treino/validação/teste, curvas de aprendizado, matriz de confusão, uma versão "que decora" para comparação e testes com a letra de pessoas novas.

## 2. Sobre o usuário

- Programa em nível iniciante e desenvolve com ajuda de IA. **Explique de forma simples**, sem jargão desnecessário, e sempre diga *por que* algo é feito.
- Precisa entender o código para explicá-lo ao professor. Comente o código em português, com comentários curtos e úteis.
- Comunique-se sempre em **português do Brasil**.

## 3. Ambiente e restrições

- **Hardware:** notebook AMD Ryzen 5 4600H, 8 GB de RAM, sem GPU NVIDIA. Todo o treino roda na CPU.
- **Sistema:** Windows (confirme com o usuário na Etapa 0). Dê os comandos para o terminal do VS Code (PowerShell).
- **Offline:** a demonstração final precisa funcionar **sem internet**. Nada pode depender de API online.
- **Memória:** CNN, webcam, interface e LLM rodam juntos em 8 GB. Prefira soluções leves.

### Tecnologias definidas

| Uso | Tecnologia |
| --- | --- |
| Linguagem | Python 3.11 com ambiente virtual `.venv` |
| CNN | TensorFlow/Keras (CPU) |
| Visão computacional | OpenCV |
| Matemática | SymPy |
| LLM local | Ollama com o modelo `qwen3:4b-instruct` (versão sem pensamento; o `qwen3:4b` comum sempre pensa, mesmo com `think=False`) |
| Agente | Python puro com a biblioteca `ollama` e *tool calling* |
| Interface | Gradio |
| Voz (opcional) | pyttsx3 (offline) |
| Gráficos | Matplotlib |
| Testes | pytest |

Não adicione outras bibliotecas pesadas (PyTorch, LangChain, Langflow etc.) sem perguntar ao usuário.

## 4. Regras invioláveis

1. **Uma etapa por vez.** Nunca comece a etapa seguinte sem o usuário escrever que aprovou a atual (por exemplo, "Etapa 2 aprovada").
2. **A CNN é treinada do zero.** Nunca use modelo pré-treinado para reconhecer os símbolos, nem baixe pesos prontos.
3. **O conjunto de teste é intocável.** Ele só é usado uma vez, na Etapa 3. Nunca o use para treinar, ajustar hiperparâmetros ou escolher modelo.
4. **Pré-processamento único.** Uma só função transforma imagens em entrada da CNN, usada tanto para o dataset quanto para a webcam. Se as duas divergirem, a rede erra mesmo bem treinada.
5. **Nunca espelhe as imagens** no data augmentation: um 2 espelhado deixa de ser um 2.
6. **Nunca use `eval()`** em expressões. Use o parser do SymPy de forma restrita.
7. **O LLM não faz contas.** Todo cálculo passa pela ferramenta do SymPy.
8. **Nunca versione o dataset** nem arquivos grandes; mantenha-os no `.gitignore`.
9. Se algo der errado ou houver dúvida sobre uma decisão do usuário, **pergunte** em vez de supor.

## 5. Como encerrar cada etapa

Ao terminar uma etapa, **pare** e entregue ao usuário, nesta ordem:

1. **O que foi feito:** 2 a 4 frases simples.
2. **Arquivos criados ou alterados.**
3. **Como testar:** os comandos exatos, prontos para copiar.
4. **Resultado esperado:** o que ele deve ver na tela se estiver tudo certo.
5. **Se der errado:** os problemas mais prováveis e o que fazer.
6. **Tarefas fora do ambiente:** tudo o que o usuário precisa fazer manualmente fora do código (instalar programas, baixar arquivos, preparar papel, testar com outras pessoas etc.). Se não houver, diga "nenhuma".
7. **Para explicar ao professor:** 1 ou 2 frases sobre o conceito da aula aplicado na etapa.

Depois, atualize o arquivo `PROGRESSO.md` com a data, a etapa, o status e os resultados (por exemplo, a acurácia). Quando o usuário aprovar a etapa, faça um commit no Git com a mensagem `Etapa N: <resumo>`.

## 6. Estrutura do projeto

```
tutor-matematica/
├── CLAUDE.md
├── PROGRESSO.md
├── README.md
├── requirements.txt
├── iniciar.bat                  # abre a aplicação com dois cliques (Etapa 9)
├── data/
│   ├── raw/                     # dataset do Kaggle extraído (fora do Git)
│   └── processed/               # dados já divididos em .npz (fora do Git)
├── models/                      # modelos .keras treinados (fora do Git)
├── reports/
│   └── figuras/                 # gráficos para a apresentação
├── src/
│   ├── config.py                # classes, caminhos, limites de confiança
│   ├── verificar_ambiente.py
│   ├── preprocessamento.py      # função única de pré-processamento
│   ├── preparar_dados.py
│   ├── modelo.py
│   ├── treinar.py
│   ├── avaliar.py
│   ├── testar_imagem.py
│   ├── segmentacao.py
│   ├── leitor_webcam.py
│   ├── ferramentas.py
│   ├── agente.py
│   ├── voz.py
│   └── app.py
└── tests/
    ├── test_preprocessamento.py
    ├── test_segmentacao.py
    └── test_ferramentas.py
```

## 7. As classes

17 classes: dígitos `0` a `9`, `+`, `-`, `×`, `÷`, `=`, `(`, `)` e a letra `x` (incógnita).

Os nomes das pastas no dataset podem ser diferentes desses símbolos (por exemplo, `times` para ×). Na Etapa 1, **liste as pastas reais** e mostre ao usuário o mapeamento proposto antes de usá-lo.

## 8. Etapas

### Etapa 0 — Ambiente

**Objetivo:** deixar o computador pronto e comprovar que tudo funciona.

- Confirmar o sistema operacional e a versão do Python.
- Criar `.venv`, `requirements.txt`, `.gitignore`, `PROGRESSO.md` e a estrutura de pastas.
- Criar `src/verificar_ambiente.py`, que checa e mostra ✅/❌ para: versão do Python, importação de TensorFlow, OpenCV, SymPy, Gradio e ollama; abertura da webcam; Ollama respondendo; modelo `qwen3:4b-instruct` instalado; dataset presente em `data/raw/`.
- Inicializar o repositório Git.

**Teste:** rodar o verificador e ver todos os itens com ✅.

**Tarefas fora do ambiente (lembre o usuário):** instalar Python 3.11, Git e Ollama; baixar o modelo com `ollama pull qwen3:4b-instruct`; criar conta no Kaggle, baixar o dataset "Handwritten Math Symbols" (kaggle.com/datasets/xainano/handwrittenmathsymbols) e extraí-lo em `data/raw/`; se o PowerShell bloquear a ativação do `.venv`, rodar `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Etapa 1 — Preparação dos dados

**Objetivo:** transformar o dataset bruto em dados limpos, balanceados e divididos.

- Listar as pastas do dataset e propor o mapeamento das 17 classes ao usuário. Esperar confirmação.
- Contar as imagens por classe e limitar cada classe a no máximo 3.000 imagens, para equilibrar.
- Criar em `src/preprocessamento.py` a função única que deixa a imagem em tons de cinza, 45x45, com a mesma polaridade do dataset (traço escuro, fundo claro, ou o inverso, conforme o dataset) e valores entre 0 e 1.
- Dividir em treino (70%), validação (15%) e teste (15%), estratificado por classe, com semente 42. Salvar em `data/processed/`.
- Gerar `reports/figuras/exemplos_por_classe.png` (grade com exemplos de cada classe) e `reports/figuras/contagem_por_classe.png`.
- Criar `tests/test_preprocessamento.py`.

**Teste:** rodar o script, conferir as contagens impressas e abrir as duas figuras.

**Para explicar ao professor:** por que separar treino, validação e teste, e por que balancear as classes.

### Etapa 2 — Treino da CNN

**Objetivo:** treinar a rede do zero e registrar o aprendizado.

- Em `src/modelo.py`, criar a CNN: 3 blocos de convolução (32, 64 e 128 filtros 3x3) com batch normalization, ReLU e max pooling; depois flatten, densa de 128 com dropout e saída softmax de 17 classes.
- Em `src/treinar.py`: data augmentation leve (rotação de até 10°, deslocamento, zoom, sem espelhar), otimizador Adam, perda categorical crossentropy, até 30 épocas, batch 64, early stopping na validação e salvamento do melhor modelo em `models/cnn_simbolos.keras`.
- Mostrar o progresso de forma clara e o tempo total.
- Salvar `reports/figuras/curvas_modelo_bom.png` (acurácia e perda de treino e validação).
- Treinar também a versão **"que decora"**: sem dropout, sem augmentation e com apenas 5% dos dados de treino. Salvar `reports/figuras/curvas_modelo_decorando.png` e uma figura com as duas versões lado a lado.

**Teste:** rodar o treino e abrir as figuras. Esperado: no modelo bom, as curvas de treino e validação ficam próximas e a validação passa de 90%; no modelo que decora, o treino vai perto de 100% e a validação fica bem abaixo.

**Para explicar ao professor:** o que é overfitting e como dropout, augmentation e early stopping o combatem.

### Etapa 3 — Avaliação

**Objetivo:** medir o resultado final de forma honesta.

- Em `src/avaliar.py`: avaliar o melhor modelo **uma única vez** no conjunto de teste. Gerar acurácia geral, relatório por classe e `reports/figuras/matriz_confusao.png`.
- Gerar `reports/figuras/erros_exemplos.png` com exemplos de imagens que a rede errou.
- Em `src/testar_imagem.py`: classificar uma foto de um símbolo isolado tirada pelo usuário, mostrando os 3 palpites mais prováveis com a confiança.

**Teste:** abrir a matriz de confusão e testar `testar_imagem.py` com fotos de símbolos escritos pelo próprio usuário.

**Tarefas fora do ambiente:** escrever alguns símbolos isolados em papel branco com caneta preta grossa e fotografá-los.

### Etapa 4 — Segmentação e leitura pela webcam

**Objetivo:** ler uma conta inteira escrita no papel.

- Em `src/segmentacao.py`: converter para preto e branco, encontrar os contornos dos símbolos, ignorar ruídos pequenos, juntar partes do mesmo símbolo (como os traços do `=` e os pontos do `÷`), ordenar da esquerda para a direita, centralizar cada recorte num quadrado e aplicar a função única de pré-processamento.
- Em `src/leitor_webcam.py`: abrir a webcam numa janela do OpenCV com um retângulo-guia; ao apertar espaço, capturar, desenhar caixas em cada símbolo com o palpite e a confiança, e imprimir a expressão montada. `Q` fecha a janela.
- Criar `tests/test_segmentacao.py` com imagens de exemplo.

**Teste:** escrever `12 + 7 × 3` no papel, mostrar à webcam e conferir a expressão lida.

**Tarefas fora do ambiente:** preparar papel branco, caneta preta grossa e boa iluminação; testar com a letra de 2 ou 3 pessoas e anotar acertos e erros.

### Etapa 5 — Ferramentas do agente

**Objetivo:** criar as ferramentas confiáveis que o agente vai usar.

Em `src/ferramentas.py`:

- `validar_expressao(simbolos)`: verifica se a expressão é válida (parênteses em par, sem operadores repetidos etc.) e explica o problema se houver.
- `resolver_ambiguidade_x(simbolos)`: decide pelo contexto se um `x` é multiplicação ou incógnita (entre dois números tende a ser ×; em expressão com `=` tende a ser incógnita).
- `calcular(expressao)`: resolve expressões aritméticas e equações do 1º grau com SymPy, sem `eval()`, devolvendo o resultado e os passos intermediários.
- `gerar_exercicio(expressao)`: cria uma conta parecida, com outros números.

Criar `tests/test_ferramentas.py` com casos como `12+7*3 = 33`, `(8-3)*4 = 20`, `2x+3=11 → x = 4` e expressões inválidas.

**Teste:** rodar `python -m pytest` e ver todos os testes passando.

### Etapa 6 — Agente com LLM local

**Objetivo:** o agente que decide, usa ferramentas, explica e confere.

Em `src/agente.py`:

- Conectar ao Ollama com `qwen3:4b-instruct` (nome lido de `config.MODELO_LLM`).
- Prompt de sistema: persona de professor paciente, respostas curtas em português, explicação passo a passo respeitando a ordem das operações, com 2 ou 3 exemplos de boas explicações (few-shot).
- Ferramentas disponíveis ao LLM: `calcular`, `perguntar_ao_visitante(simbolo, opcoes)` e `gerar_exercicio`.
- **Regra no código, não só no prompt:** se algum símbolo tiver confiança abaixo do limite definido em `config.py`, o agente pergunta antes de calcular.
- **Verificação (Chain of Verification):** antes de responder, conferir se o resultado citado na explicação é igual ao da ferramenta; se não for, gerar a explicação de novo.
- Manter o histórico da conversa para perguntas de acompanhamento.
- Permitir teste pelo terminal com uma expressão e confianças simuladas.

**Teste:** rodar pelo terminal com `12+7*3` e com um caso de confiança baixa, e conferir a explicação e a pergunta. Medir o tempo de resposta.

**Para explicar ao professor:** o que torna isso um agente (perceber, decidir, usar ferramentas, agir) e por que o LLM não faz a conta sozinho.

### Etapa 7 — Interface

**Objetivo:** juntar tudo numa interface simples e bem apresentável.

Em `src/app.py`, com Gradio, rodando em `localhost`:

- **Aba "Webcam":** imagem da câmera, botão "Ler conta", imagem com as caixas e palpites, expressão lida e chat com o agente.
- **Aba "Desenhar" (plano B):** quadro para desenhar a conta com o mouse, usando o mesmo fluxo.
- Visual limpo, título do projeto, fontes legíveis à distância e indicação da confiança da CNN.
- Carregar o modelo e o LLM uma vez só, na abertura.

**Teste:** abrir a interface no navegador e fazer o ciclo completo nas duas abas.

### Etapa 8 — Voz (opcional)

**Objetivo:** o agente fala as respostas.

- Em `src/voz.py`, usar pyttsx3 com uma voz em português, se houver no Windows.
- Botão na interface para ligar e desligar a voz, sem travar a tela enquanto fala.

**Teste:** ouvir a explicação de uma conta.

**Tarefas fora do ambiente:** se não houver voz em português, instalá-la nas configurações de fala do Windows.

### Etapa 9 — Ensaio final

**Objetivo:** garantir que tudo funcione na feira.

- Criar `iniciar.bat`, que ativa o ambiente e abre a aplicação.
- Rodar o fluxo completo com o Wi-Fi desligado e medir o uso de RAM.
- Se a RAM apertar, oferecer a troca do LLM por um modelo de 1B (como `gemma3:1b`).
- Escrever o `README.md` com: o que é o projeto, como rodar, arquitetura, resultados (acurácia, matriz de confusão, teste com pessoas) e limitações.
- Conferir se todas as figuras de `reports/figuras/` estão prontas para a apresentação.

**Tarefas fora do ambiente:** testar com 10 pessoas diferentes e anotar os acertos; montar banner ou slides; ensaiar a explicação; separar papel, canetas e uma luminária para o estande.

## 9. Escopo das contas

- **Nível 1 (obrigatório):** expressões com inteiros, `+ − × ÷`, parênteses e ordem das operações. Ex.: `12 + 7 × 3`, `(8 − 3) × 4`.
- **Nível 2 (se houver tempo):** equações do 1º grau com uma incógnita, com multiplicação escrita colada (`2x`). Ex.: `2x + 3 = 11`.
- **Fora do escopo:** frações empilhadas, potências, raízes e decimais, porque a leitura é em linha, da esquerda para a direita.
