"""Etapa 8: o tutor fala as respostas (pyttsx3, offline, voz do próprio Windows).

Por que um processo separado?
  O pyttsx3 TRAVA o programa enquanto fala e não gosta de ser usado por várias
  linhas de execução (como as do Gradio). Então cada fala roda num pequeno
  Python à parte: a tela continua livre e, para parar no meio, basta encerrar
  esse processo.

Teste pelo terminal:
    python src/voz.py                      # fala uma explicação de exemplo
    python src/voz.py "7 × 3 = 21"         # fala o texto dado
    python src/voz.py --listar             # mostra as vozes instaladas
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

VELOCIDADE = 170          # palavras por minuto (o padrão, 200, é rápido demais para crianças)
IDIOMA = "pt"             # prefere uma voz em português (no Windows: "Microsoft Maria")
EXEMPLO = ("Primeiro fazemos a multiplicação, porque ela vem antes da soma: 7 × 3 = 21. "
           "A conta fica 12 + 21. Agora somamos: 12 + 21 = 33. O resultado é 33!")


# ---------------------------------------------------------------------------
# 1. Preparar o texto para ser falado
# ---------------------------------------------------------------------------

def texto_para_fala(texto):
    """Troca símbolos por palavras, para a voz ler a conta do jeito certo.

    "7 × 3 = 21" -> "7 vezes 3 é igual a 21";  "2x = 8" -> "2 xis é igual a 8".
    Também tira negrito, emojis e outros símbolos que a voz leria em voz alta."""
    t = texto.replace("**", "").replace("__", "").replace("`", "")
    t = re.sub(r"[\U0001F000-\U0001FFFF☀-➿⬀-⯿️]", "", t)   # emojis, ✔, ✅
    t = t.replace("−", "-").replace("–", "-")
    t = re.sub(r"(\d)\s*/\s*(\d)", r"\1 sobre \2", t)                     # 7/2 -> 7 sobre 2
    t = re.sub(r"(\d)\s*x\b", r"\1 xis", t)                                # 2x -> 2 xis
    t = re.sub(r"(?<![\wÀ-ú])x(?![\wÀ-ú])", "xis", t)                      # x sozinho -> xis
    t = re.sub(r"\?(?=\S)", " interrogação ", t)                          # 12+?×3 (símbolo em dúvida)
    t = re.sub(r"\(\s*(?=[\d-]|xis)", " abre parênteses, ", t)             # ( antes de número
    t = re.sub(r"(?:(?<=\d)|(?<=xis))\s*\)", ", fecha parênteses ", t)     # ) depois de número/xis
    t = re.sub(r"(?<![\wÀ-ú])-\s*(?=\d|xis|abre)", " menos ", t)            # -3, 8 - 3
    t = re.sub(r"(?:(?<=\d)|(?<=xis)|(?<=parênteses ))\s*-\s*", " menos ", t)
    for simbolo, palavra in (("×", " vezes "), ("*", " vezes "), ("÷", " dividido por "), ("+", " mais "),
                             ("≠", " é diferente de "), ("=", " é igual a ")):
        t = t.replace(simbolo, palavra)
    t = re.sub(r"[#>|]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
# 2. Falar (dentro do processo separado)
# ---------------------------------------------------------------------------

def escolher_voz(vozes, idioma=IDIOMA):
    """A primeira voz no idioma pedido (pelo código, ex.: pt-BR, ou pelo nome)."""
    for v in vozes:
        idiomas = [str(i).lower() for i in (getattr(v, "languages", None) or [])]
        if any(i.startswith(idioma) for i in idiomas) or "portugu" in v.name.lower():
            return v
    return None


def falar_agora(texto):
    """Fala o texto e só volta quando terminar (por isso roda no processo separado)."""
    import pyttsx3

    motor = pyttsx3.init()
    voz = escolher_voz(motor.getProperty("voices"))
    if voz is not None:
        motor.setProperty("voice", voz.id)
    motor.setProperty("rate", VELOCIDADE)
    motor.say(texto_para_fala(texto))
    motor.runAndWait()


# ---------------------------------------------------------------------------
# 3. Controle usado pela interface (não trava a tela)
# ---------------------------------------------------------------------------

class Voz:
    """Fala em segundo plano. falar() volta na hora; parar() interrompe no meio."""

    def __init__(self, comando=None):
        # comando que fala o texto recebido pela entrada padrão (trocável nos testes)
        self.comando = comando or [sys.executable, str(Path(__file__).resolve()), "--entrada"]
        self.processo = None

    def falar(self, texto):
        self.parar()                               # uma fala nova interrompe a anterior
        if not texto or not texto.strip():
            return
        self.processo = subprocess.Popen(self.comando, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            self.processo.stdin.write(texto.encode("utf-8"))
            self.processo.stdin.close()            # fecha a entrada: o processo começa a falar
        except OSError:
            pass

    def falando(self):
        return self.processo is not None and self.processo.poll() is None

    def parar(self):
        if self.falando():
            self.processo.terminate()              # encerra o processo: a fala para na hora
            try:
                self.processo.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.processo.kill()
        self.processo = None


def descrever_voz():
    """Nome da voz que será usada (para mostrar ao abrir o programa), ou None."""
    try:
        import pyttsx3

        motor = pyttsx3.init()
        voz = escolher_voz(motor.getProperty("voices"))
        return voz.name if voz else "voz padrão do Windows (não há voz em português)"
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Faz o tutor falar um texto.")
    parser.add_argument("texto", nargs="?", default=EXEMPLO)
    parser.add_argument("--entrada", action="store_true", help="lê o texto da entrada padrão (uso interno)")
    parser.add_argument("--listar", action="store_true", help="mostra as vozes instaladas")
    args = parser.parse_args()

    if args.entrada:
        falar_agora(sys.stdin.buffer.read().decode("utf-8"))
        return
    sys.stdout.reconfigure(encoding="utf-8")
    if args.listar:
        import pyttsx3

        for v in pyttsx3.init().getProperty("voices"):
            print(f"- {v.name}  {v.languages}")
        return
    print(f"Voz: {descrever_voz()}")
    print(f"Texto falado: {texto_para_fala(args.texto)}")
    falar_agora(args.texto)


if __name__ == "__main__":
    main()
