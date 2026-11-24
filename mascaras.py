"""Etapas 2 e 3: máscara de cor em HSV + limpeza morfológica.

Funções reutilizadas pelo calibrador, pela demo e, depois, pelo Mean Shift.
"""
import json
from pathlib import Path

import cv2
import numpy as np

CAMINHO_CONFIG = Path(__file__).with_name("config_cores.json")
LUVAS = ("verde", "magenta")


def carregar_config(caminho=CAMINHO_CONFIG):
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def salvar_config(config, caminho=CAMINHO_CONFIG):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def mascara_hsv(hsv, faixas):
    """Une (OR) as máscaras de uma ou mais faixas HSV.

    Mais de uma faixa é útil quando a cor cruza o 'corte' do H (0/180),
    como acontece com vermelhos e alguns magentas.
    """
    mascara = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for faixa in faixas:
        lo = np.array(faixa["min"], dtype=np.uint8)
        hi = np.array(faixa["max"], dtype=np.uint8)
        mascara |= cv2.inRange(hsv, lo, hi)
    return mascara


def limpar_mascara(mascara, tamanho_kernel=5, iter_erosao=1, iter_dilatacao=2):
    """Erosão remove pontinhos de ruído; dilatação recupera o corpo da mão."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (tamanho_kernel, tamanho_kernel)
    )
    m = cv2.erode(mascara, kernel, iterations=iter_erosao)
    m = cv2.dilate(m, kernel, iterations=iter_dilatacao)
    return m


def preprocessar(frame_bgr, desfoque=5):
    """Leve desfoque antes do HSV reduz ruído de sensor da webcam."""
    if desfoque > 1:
        frame_bgr = cv2.GaussianBlur(frame_bgr, (desfoque, desfoque), 0)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)


def mascaras_luvas(frame_bgr, config):
    """Retorna {luva: (mascara_bruta, mascara_limpa)} para cada luva."""
    hsv = preprocessar(frame_bgr)
    morf = config["morfologia"]
    resultado = {}
    for luva in LUVAS:
        bruta = mascara_hsv(hsv, config[luva]["faixas"])
        limpa = limpar_mascara(bruta, **morf)
        resultado[luva] = (bruta, limpa)
    return resultado


def pixels_candidatos(mascara):
    """Coordenadas (x, y) dos pixels ativos — entrada do Mean Shift (Etapa 4)."""
    ys, xs = np.nonzero(mascara)
    return np.column_stack((xs, ys)).astype(np.float64)
