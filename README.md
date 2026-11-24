# Jogo de dança com Mean Shift

Trabalho de Cálculo II: um jogo de detecção de movimento no estilo "dance game"
(coreografia da Macarena) em que a posição de cada mão é encontrada pelo
algoritmo de **Mean Shift**, implementado do zero com NumPy (estimativa de
densidade por kernel gaussiano + subida de gradiente até a moda).

Cada mão usa uma luva de cor sólida (verde-neon e magenta), detectada por
limiar em HSV e limpa com morfologia (erosão + dilatação).

## Estado atual

- [x] Etapa 1 – captura da webcam (`captura.py`)
- [x] Etapa 2 – calibração HSV das duas cores (`calibrar_hsv.py`)
- [x] Etapa 3 – limpeza morfológica (`mascaras.py`)
- [ ] Etapa 4 – núcleo do Mean Shift
- [ ] Etapa 5 – visualização (densidade, curvas de nível, gradiente)
- [ ] Etapa 6 – roteiro da coreografia sincronizado com a música
- [ ] Etapa 7 – pontuação e placar

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Como usar

1. **Calibrar as cores** (com a luva/objeto na frente da câmera, na luz em que
   vai jogar). Ajuste os controles até só o objeto ficar branco na máscara e
   aperte `s` para salvar em `config_cores.json` (`q` sai):

   ```bash
   python calibrar_hsv.py verde
   python calibrar_hsv.py magenta
   ```

2. **Testar a captura e as máscaras**:

   ```bash
   python captura.py
   ```

   Teclas: `q` sai, `m` mostra/esconde as máscaras (antes | depois da limpeza),
   `p` salva prints.

> A calibração depende da iluminação: cada pessoa provavelmente vai precisar
> recalibrar no próprio ambiente.

## Arquivos

| Arquivo | Função |
|---|---|
| `mascaras.py` | Limiar HSV, limpeza morfológica e extração dos pixels candidatos |
| `calibrar_hsv.py` | Calibrador interativo com trackbars |
| `captura.py` | Webcam + contornos das luvas + máscaras antes/depois |
| `config_cores.json` | Faixas HSV calibradas e parâmetros da morfologia |
