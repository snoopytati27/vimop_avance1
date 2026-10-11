"""
analysis/wheels.py — WheelLocator: puntos de contacto de las ruedas con el piso.

    locator = WheelLocator(cfg, device, half)
    locator.locate(frame, det) -> {"rear": (x, y), "front": (x, y), "source": ..., "confidence": ...}

Backends (cfg.WHEEL_BACKEND):
    "pose"  YOLO-pose con 2 puntos clave (rueda trasera, rueda delantera) en cfg.WHEEL_MODEL_PATH.
    "mask"  respaldo sin entrenar: franja inferior de la máscara de segmentación, eje principal (PCA)
            y extremos metidos INSET_FRACTION hacia adentro.
    "auto"  "pose" si existe el archivo del modelo; si no, "mask".

`source` de cada resultado:
    "pose"  puntos del modelo entrenado
    "mask"  estimados con la máscara (aproximado). El sentido trasera/delantera NO se distingue:
            R y F son los dos extremos, de izquierda a derecha
    "bbox"  sin modelo ni máscara: solo un segmento a lo largo del bbox (la orientación no es confiable)

Si un punto del modelo no llega a WHEEL_KPT_CONF, esa moto usa el respaldo "mask".
R y F se suavizan con una media móvil por track (deque) para que la línea no tiemble.
"""
from __future__ import annotations

import logging
from collections import deque

import numpy as np

from analysis.geometry import bbox_axis, wheel_axis_from_mask
from core.detector import Detection
from utils import bbox_iou

log = logging.getLogger("vimop.wheels")

SRC_POSE, SRC_MASK, SRC_BBOX = "pose", "mask", "bbox"
SOURCE_LABELS = {SRC_POSE: "modelo entrenado", SRC_MASK: "máscara (aproximado)", SRC_BBOX: "bbox (sin ruedas)"}
MAX_TRACKS_KEPT = 256
MIN_POSE_IOU = 0.30


class WheelLocator:
    def __init__(self, cfg, device: str = "cpu", half: bool = False):
        self.cfg = cfg
        self.device = device
        self.half = bool(half) and str(device).startswith("cuda")
        self.model = None
        self.model_path = cfg.resolve_path(cfg.WHEEL_MODEL_PATH)
        backend = str(cfg.WHEEL_BACKEND).lower()
        if backend not in ("pose", "mask", "auto"):
            log.warning("WHEEL_BACKEND '%s' desconocido; se usa 'auto'.", backend)
            backend = "auto"
        self.requested = backend
        if backend in ("pose", "auto"):
            self._load_pose(warn=backend == "pose")
        self.backend = SRC_POSE if self.model is not None else SRC_MASK
        self._history: dict[int, deque] = {}
        self._cache_key = None
        self._cache: list[tuple[tuple, np.ndarray, np.ndarray]] = []
        log.info(self.description)

    # ------------------------------------------------------------------
    def _load_pose(self, warn: bool) -> None:
        if not self.model_path.exists():
            if warn:
                log.warning("WHEEL_BACKEND='pose' pero no existe %s: se usa el respaldo 'mask'.", self.model_path)
            return
        try:
            from ultralytics import YOLO
            model = YOLO(str(self.model_path))
            if getattr(model, "task", "") != "pose":
                raise ValueError(f"el modelo es de tipo '{getattr(model, 'task', '?')}', se esperaba 'pose'")
            self.model = model
        except Exception as exc:                           # modelo dañado, versión distinta, sin ultralytics...
            log.warning("No se pudo cargar %s (%s): se usa el respaldo 'mask'.", self.model_path.name, exc)

    @property
    def description(self) -> str:
        """Texto para la interfaz."""
        if self.model is not None:
            return "Ruedas: modelo entrenado"
        return "Ruedas: estimadas por máscara (aproximado)"

    @property
    def has_model(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    def locate(self, frame: np.ndarray | None, det: Detection) -> dict:
        found = None
        if self.model is not None and frame is not None:
            found = self._from_pose(frame, det)
        if found is None:
            found = self._from_mask(det)
        if found is None:
            r, f = bbox_axis(det.bbox, float(self.cfg.BOTTOM_BAND), float(self.cfg.INSET_FRACTION))
            found = {"rear": r, "front": f, "source": SRC_BBOX, "confidence": 0.0}
        return self._smooth(det.track_id, found)

    # ---- backend pose --------------------------------------------------
    def _pose_detections(self, frame: np.ndarray):
        """Una inferencia de pose por frame; se reutiliza para todas las motos del mismo frame."""
        key = (id(frame), frame.shape, int(frame[::61, ::67].sum()))
        if key == self._cache_key:
            return self._cache
        out = []
        res = self.model.predict(frame, imgsz=int(self.cfg.WHEEL_IMG_SIZE), conf=0.20, device=self.device,
                                 half=self.half, verbose=False)[0]
        if res.boxes is not None and len(res.boxes) and res.keypoints is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            kxy = res.keypoints.xy.cpu().numpy()                          # (n, 2, 2)
            kconf = (res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None
                     else np.ones(kxy.shape[:2], dtype=np.float32))
            for i in range(len(boxes)):
                if kxy.shape[1] >= 2:
                    out.append((tuple(float(v) for v in boxes[i]), kxy[i], kconf[i]))
        self._cache_key, self._cache = key, out
        return out

    def _from_pose(self, frame: np.ndarray, det: Detection) -> dict | None:
        best, best_iou = None, MIN_POSE_IOU
        for box, kxy, kconf in self._pose_detections(frame):
            iou = bbox_iou(box, det.bbox)
            if iou >= best_iou:
                best, best_iou = (kxy, kconf), iou
        if best is None:
            return None
        kxy, kconf = best
        if float(kconf[0]) < float(self.cfg.WHEEL_KPT_CONF) or float(kconf[1]) < float(self.cfg.WHEEL_KPT_CONF):
            return None
        return {"rear": np.asarray(kxy[0], dtype=np.float64), "front": np.asarray(kxy[1], dtype=np.float64),
                "source": SRC_POSE, "confidence": float(min(kconf[0], kconf[1]))}

    # ---- backend mask --------------------------------------------------
    def _from_mask(self, det: Detection) -> dict | None:
        if det.mask is None or not det.mask.any():
            return None
        axis = wheel_axis_from_mask(det.mask, (det.bbox[0], det.bbox[1]), float(self.cfg.BOTTOM_BAND),
                                    float(self.cfg.INSET_FRACTION))
        if axis is None:
            return None
        return {"rear": axis[0], "front": axis[1], "source": SRC_MASK, "confidence": 0.5}

    # ---- suavizado -------------------------------------------------------
    def _smooth(self, track_id: int | None, res: dict) -> dict:
        n = int(self.cfg.WHEEL_SMOOTH_FRAMES)
        if track_id is None or n <= 1:
            return res
        hist = self._history.get(track_id)
        r, f = np.asarray(res["rear"], dtype=np.float64), np.asarray(res["front"], dtype=np.float64)
        if hist is None:
            if len(self._history) >= MAX_TRACKS_KEPT:
                self._history.pop(next(iter(self._history)))
            hist = self._history[track_id] = deque(maxlen=n)
        elif hist:
            pr = np.mean([h[0] for h in hist], axis=0)
            pf = np.mean([h[1] for h in hist], axis=0)
            straight = np.linalg.norm(r - pr) + np.linalg.norm(f - pf)
            swapped = np.linalg.norm(r - pf) + np.linalg.norm(f - pr)
            if swapped < straight and res["source"] != SRC_POSE:   # mask/bbox: extremos sin sentido propio
                r, f = f, r
        hist.append((r, f))
        out = dict(res)
        out["rear"] = np.mean([h[0] for h in hist], axis=0)
        out["front"] = np.mean([h[1] for h in hist], axis=0)
        return out
