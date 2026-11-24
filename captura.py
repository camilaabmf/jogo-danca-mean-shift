"""Etapas 1 e 3: captura da webcam + máscaras das duas luvas, antes/depois da limpeza.

Uso:  python captura.py
Teclas:  q = sair   m = alterna exibição das máscaras   p = salva print (para os slides)
"""
import sys
import time

import cv2
import numpy as np

from mascaras import carregar_config, mascaras_luvas

LARGURA, ALTURA = 640, 480
COR_DESENHO = {"verde": (0, 255, 0), "magenta": (255, 0, 255)}


def rotular(img, texto):
    img = img.copy()
    cv2.putText(img, texto, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


def main():
    config = carregar_config()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Não foi possível abrir a webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA)

    mostrar_mascaras = True
    fps, t_anterior = 0.0, time.perf_counter()
    n_print = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Falha ao ler quadro.")
            break
        frame = cv2.flip(frame, 1)  # espelho: mais natural para dançar

        mascaras = mascaras_luvas(frame, config)

        # Contorno da área detectada de cada luva sobre o vídeo
        visual = frame.copy()
        for luva, (_, limpa) in mascaras.items():
            contornos, _ = cv2.findContours(limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(visual, contornos, -1, COR_DESENHO[luva], 2)

        # FPS suavizado (média móvel exponencial)
        agora = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(agora - t_anterior, 1e-6))
        t_anterior = agora
        cv2.putText(visual, f"FPS: {fps:4.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Webcam", visual)

        if mostrar_mascaras:
            linhas = []
            for luva, (bruta, limpa) in mascaras.items():
                antes = rotular(cv2.cvtColor(bruta, cv2.COLOR_GRAY2BGR), f"{luva}: bruta")
                depois = rotular(cv2.cvtColor(limpa, cv2.COLOR_GRAY2BGR), f"{luva}: limpa")
                linhas.append(np.hstack((antes, depois)))
            painel = cv2.resize(np.vstack(linhas), None, fx=0.5, fy=0.5)
            cv2.imshow("Mascaras (antes | depois)", painel)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("m"):
            mostrar_mascaras = not mostrar_mascaras
            if not mostrar_mascaras:
                cv2.destroyWindow("Mascaras (antes | depois)")
        if tecla == ord("p"):
            n_print += 1
            cv2.imwrite(f"print_webcam_{n_print}.png", visual)
            if mostrar_mascaras:
                cv2.imwrite(f"print_mascaras_{n_print}.png", painel)
            print(f"Prints {n_print} salvos.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
