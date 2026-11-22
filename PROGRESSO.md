# Progresso do projeto

| Data | Etapa | Status | Resultados |
| --- | --- | --- | --- |
| 2026-11-22 | Etapa 0 — Ambiente | ✅ Aprovada | 10/10 itens ✅ no verificador. Python 3.11.0, TensorFlow 2.21.0 (CPU), OpenCV 5.1.0, SymPy 1.14.0, Gradio 6.32.0, ollama 0.6.3, webcam 640x480, dataset com 82 pastas. LLM trocado de `qwen3:4b` para `qwen3:4b-instruct`: o `qwen3:4b` atual sempre "pensa" e o raciocínio vazava na resposta mesmo com `think=False`. Com o instruct, a resposta veio limpa em ~9 s (incluindo o carregamento do modelo). |
| 2026-11-22 | Etapa 1 — Preparação dos dados | ✅ Aprovada | Mapeamento das 17 classes aprovado (`times`→×, `div`→÷, `X`→x). **77% do dataset eram cópias exatas** (174.176 de 225.832 arquivos; cada imagem aparecia até 11 vezes) e foram removidas, para o teste não conter imagens vistas no treino. Restaram 34.722 imagens únicas: treino 24.305, validação 5.209, teste 5.208 (estratificado, semente 42). Classes menores: ÷ 157, × 600, 8 731. Sobreposição entre partes: 0. Pré-processamento único com afinamento do traço (Zhang-Suen). 10/10 testes passando. |
