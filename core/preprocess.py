"""
core/preprocess.py — Preprocesado de imagen COMPARTIDO por la aplicación y por tools/label_wheels.py.

Así el dataset de ruedas se etiqueta y se entrena con las mismas imágenes que verá el programa.

    frame = prepare_frame(frame, cfg)

Hoy hace una sola cosa opcional: con IR_CLAHE = True, las imágenes de cámara infrarroja (visión nocturna,
casi sin color) se ecualizan localmente (CLAHE sobre la luminancia) para que el detector vea mejor los contornos.
Con IR_CLAHE = False (por defecto) la imagen no se toca.
"""
from __future__ import annotations

import cv2
import numpy as np


def is_infrared(frame: np.ndarray, max_saturation: float = 12.0) -> bool:
    """Imagen casi sin color (modo infrarrojo de las cámaras de seguridad)."""
    if frame.ndim == 2 or frame.shape[2] == 1:
        return True
    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) < float(max_saturation)


def apply_ir_clahe(frame: np.ndarray, clip_limit: float = 2.5, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def prepare_frame(frame: np.ndarray, cfg) -> np.ndarray:
    """Preprocesado antes de detectar (misma función en el programa y en la herramienta de etiquetado)."""
    if bool(getattr(cfg, "IR_CLAHE", False)) and is_infrared(frame, float(cfg.IR_SATURATION_MAX)):
        return apply_ir_clahe(frame)
    return frame
