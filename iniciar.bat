@echo off
REM ==========================================================================
REM  Tutor de Matematica que Le sua Letra - abre tudo com dois cliques
REM
REM  1. Vai para a pasta do projeto (onde este arquivo esta)
REM  2. Confere o ambiente virtual (.venv)
REM  3. Abre o Ollama, se estiver fechado, e espera ele responder
REM  4. Abre a interface no navegador (http://localhost:7860)
REM ==========================================================================
chcp 65001 >nul
cd /d "%~dp0"
title Tutor de Matematica
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set GRADIO_ANALYTICS_ENABLED=False

echo.
echo  ===  Tutor de Matematica que Le sua Letra  ===
echo.

if not exist ".venv\Scripts\python.exe" (
    echo  [ERRO] Nao achei o ambiente virtual .venv nesta pasta.
    echo         Crie com:  python -m venv .venv
    echo         e instale: .venv\Scripts\pip install -r requirements.txt
    goto erro
)

REM --- Ollama: ja esta respondendo? ---------------------------------------
ollama list >nul 2>&1
if not errorlevel 1 goto ollama_ok

echo  O Ollama estava fechado. Abrindo o Ollama...
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" (
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
) else (
    start "Ollama" /min ollama serve
)
echo  Esperando o Ollama ficar pronto (ate 60 segundos)...
REM Espera 1 segundo por volta com "ping" (funciona em qualquer Windows; o comando
REM "timeout" pode ser trocado por outro programa com o mesmo nome, como o do Git)
for /l %%i in (1,1,60) do (
    ping -n 2 127.0.0.1 >nul
    ollama list >nul 2>&1 && goto ollama_aberto
)
echo  [AVISO] O Ollama nao respondeu. O tutor vai abrir mesmo assim,
echo          mas vai usar a explicacao automatica (sem o LLM).
echo.
goto abrir_tutor

:ollama_aberto
echo  Ollama pronto.
echo  Como ele acabou de abrir, o modelo de linguagem ainda vai ser carregado
echo  do disco: desta vez o tutor leva cerca de 1 minuto para abrir.
echo.

:ollama_ok
:abrir_tutor
echo  Carregando a rede neural, o LLM e a voz...
echo  O navegador abre sozinho. Para FECHAR o tutor, feche esta janela.
echo  (Tudo o que aparece aqui fica salvo em iniciar.log)
echo.
".venv\Scripts\python.exe" src\app.py
if errorlevel 1 goto erro
exit /b 0

:erro
echo.
echo  Algo deu errado. A mensagem acima tambem ficou salva em iniciar.log.
echo  Aperte uma tecla para fechar.
pause >nul
exit /b 1
