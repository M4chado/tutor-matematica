# Testes de leitura com pessoas

Como testar: a pessoa escreve a conta no papel (canetinha preta grossa, papel liso), você
mostra à webcam na interface (`iniciar.bat`, aba Webcam) e clica em "Ler conta".
As fotos de cada leitura ficam salvas em `capturas/` (fora do Git).

- **Símbolos certos:** quantos símbolos foram lidos corretamente, de quantos havia.
  O `×` lido no lugar do `x` (ou o contrário) conta como certo, porque o tutor decide pelo contexto.
- **O que o tutor fez:** calculou, perguntou (símbolo duvidoso) ou avisou que a conta parecia mal escrita.

Use "Pessoa 1, 2, 3..." em vez de nomes, para preservar a privacidade de quem testou.

## Resultados

| Data | Pessoa | Conta escrita | Conta lida | Símbolos certos | O que o tutor fez | Observações |
| --- | --- | --- | --- | --- | --- | --- |
| 23/11/2026 | Pessoa 1 (autor) | `4x+5=21` | `4×+5=2)` | 6/7 | avisou: ")" sem "(" | `×` → `x` pelo contexto ✔; o `1` com gancho foi lido como `)` (98%) |
| 23/11/2026 | Pessoa 1 (autor) | `4x+5=21` | `+5=2)` | 4/7 | avisou: ")" sem "(" | `4x` encostado numa fronteira falsa da folha (luz fraca) e descartado; `1` → `)` |
| 23/11/2026 | Pessoa 1 (autor) | `4x+5=21` | `4×+5=2)` | 6/7 | perguntou: "+ ou 7?" e ") ou 1?" | `1` → `)` só com 58%: duvidoso, então o tutor perguntou |
| 23/11/2026 | Pessoa 1 (autor) | `4+5×2` | `475×2` | 4/5 | perguntou: "7 ou +?" | `+` lido como `7` com 70%: duvidoso, então o tutor perguntou |
| | Pessoa 2 | | | / | | |
| | Pessoa 3 | | | / | | |
| | Pessoa 4 | | | / | | |
| | Pessoa 5 | | | / | | |
| | Pessoa 6 | | | / | | |
| | Pessoa 7 | | | / | | |
| | Pessoa 8 | | | / | | |
| | Pessoa 9 | | | / | | |
| | Pessoa 10 | | | / | | |

**Total até agora:** 20 de 26 símbolos certos (77%), só da Pessoa 1, à noite, com luz fraca e webcam do notebook.

**Em nenhuma das 4 contas o tutor deu uma resposta errada:** em 2 ele perguntou sobre o símbolo
duvidoso e em 2 avisou que a conta parecia mal escrita e pediu para conferir.

**Erro mais comum:** o `1` escrito com "gancho" no alto é lido como `)`. É a mesma confusão
`1` ↔ `(` / `)` vista na matriz de confusão da Etapa 3.

Releitura das mesmas 4 fotos com a versão final do programa: 21 de 26 (81%). Na 2ª foto o
`x` passou a ser encontrado; o `4` continua de fora.
