"""Etapa 2: calibrador HSV com trackbars.

Uso:  python calibrar_hsv.py verde      (ou magenta)
Teclas:  s = salvar no config_cores.json   q = sair sem salvar
Dica: vista a luva, ponha-a na frente da câmera e aperte os limites até
que SÓ ela fique branca na máscara. Calibre na mesma luz da apresentação.
"""
import sys

import cv2
import numpy as np

from mascaras import LUVAS, carregar_config, limpar_mascara, preprocessar, salvar_config

JANELA = "Calibracao HSV"


def main():
    luva = sys.argv[1] if len(sys.argv) > 1 else "verde"
    if luva not in LUVAS:
        sys.exit(f"Luva deve ser uma de {LUVAS}")

    config = carregar_config()
    faixa = config[luva]["faixas"][0]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Não foi possível abrir a webcam.")

    cv2.namedWindow(JANELA)
    for i, canal in enumerate("HSV"):
        maximo = 179 if canal == "H" else 255
        cv2.createTrackbar(f"{canal} min", JANELA, faixa["min"][i], maximo, lambda v: None)
        cv2.createTrackbar(f"{canal} max", JANELA, faixa["max"][i], maximo, lambda v: None)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        hsv = preprocessar(frame)

        lo = [cv2.getTrackbarPos(f"{c} min", JANELA) for c in "HSV"]
        hi = [cv2.getTrackbarPos(f"{c} max", JANELA) for c in "HSV"]
        mascara = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        limpa = limpar_mascara(mascara, **config["morfologia"])

        resultado = cv2.bitwise_and(frame, frame, mask=limpa)
        mascara_bgr = cv2.cvtColor(limpa, cv2.COLOR_GRAY2BGR)
        painel = np.hstack((frame, mascara_bgr, resultado))
        painel = cv2.resize(painel, None, fx=0.6, fy=0.6)
        cv2.putText(painel, f"Luva: {luva}  [s]alvar  [q]uit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(JANELA, painel)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("s"):
            config[luva]["faixas"][0] = {"min": lo, "max": hi}
            salvar_config(config)
            print(f"Salvo para '{luva}': min={lo} max={hi}")
        elif tecla == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
