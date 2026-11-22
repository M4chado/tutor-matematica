# Testes de leitura com pessoas

Como testar: a pessoa escreve a conta no papel (caneta preta grossa), você mostra à
webcam (`python src/leitor_webcam.py`), aperta ESPAÇO e anota o resultado.
As fotos de cada leitura ficam salvas em `capturas/`.

"Símbolos certos" = quantos símbolos foram lidos corretamente, de quantos havia.

| Data | Pessoa | Conta escrita | Conta lida | Símbolos certos | Observações |
| --- | --- | --- | --- | --- | --- |
| | Pessoa 1 (eu) | `12+7×3` | | /6 | |
| | Pessoa 1 (eu) | `(8-3)×4` | | /7 | |
| | Pessoa 2 | `12+7×3` | | /6 | |
| | Pessoa 2 | `(8-3)×4` | | /7 | |
| | Pessoa 3 | `12+7×3` | | /6 | |
| | Pessoa 3 | `(8-3)×4` | | /7 | |

**Total:** ___ de ___ símbolos certos (___%).

Use "Pessoa 1, 2, 3" em vez de nomes, para preservar a privacidade de quem testou.
