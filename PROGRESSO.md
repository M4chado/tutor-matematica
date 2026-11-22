# Progresso do projeto

| Data | Etapa | Status | Resultados |
| --- | --- | --- | --- |
| 2026-11-22 | Etapa 0 — Ambiente | ✅ Aprovada | 10/10 itens ✅ no verificador. Python 3.11.0, TensorFlow 2.21.0 (CPU), OpenCV 5.1.0, SymPy 1.14.0, Gradio 6.32.0, ollama 0.6.3, webcam 640x480, dataset com 82 pastas. LLM trocado de `qwen3:4b` para `qwen3:4b-instruct`: o `qwen3:4b` atual sempre "pensa" e o raciocínio vazava na resposta mesmo com `think=False`. Com o instruct, a resposta veio limpa em ~9 s (incluindo o carregamento do modelo). |
