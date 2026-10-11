"""
core/detector.py — Detección + tracking con Ultralytics YOLO.

    detector = YoloDetector(cfg, device, is_gpu)
    batch = detector.track(frame)        # video/cámara: model.track(persist=True, tracker=bytetrack/botsort)
    batch = detector.detect(frame)       # foto: model.predict, una sola vez, sin tracking temporal
    batch.motorcycles  -> [Detection(track_id, bbox, mask, confidence, class_name)]
    batch.persons      -> [Detection(...)]

Una sola inferencia sirve para los dos "detectores" (motos y personas): correr
dos redes sobre el mismo frame duplicaría el costo sin aportar nada.

Por defecto se usa un detector con segmentación (models/yolo11s-seg.pt): cada Detection trae además
su máscara (bool, recortada al bbox), que alimenta el respaldo "mask" de WheelLocator (línea de ejes).
Con un modelo sin segmentación (yolo11s.pt / yolo11m.pt) solo hay bounding box.

El tracker se configura con un YAML generado a partir del oficial de Ultralytics
(bytetrack.yaml / botsort.yaml) con los umbrales de config.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("vimop.detector")


class ModelNotFoundError(RuntimeError):
    """El modelo YOLO no existe, no se pudo descargar o no tiene la clase 'motorcycle'."""


@dataclass
class Detection:
    kind: str                                  # "motorcycle" | "person"
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]            # x1, y1, x2, y2 (dentro del frame)
    mask: np.ndarray | None = None             # bool (y2-y1, x2-x1)
    track_id: int | None = None                # ID del tracker de Ultralytics

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def diag(self) -> float:
        return float(np.hypot(self.width, self.height))

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, float(self.bbox[3]))

    def mask_or_box(self) -> np.ndarray:
        if self.mask is not None and self.mask.any():
            return self.mask
        return np.ones((self.height, self.width), dtype=bool)


@dataclass
class DetectionBatch:
    motorcycles: list[Detection] = field(default_factory=list)
    persons: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0


def select_device(preference: str = "auto") -> tuple[str, bool, str]:
    """CUDA disponible -> 'cuda:0' (GPU); si no -> 'cpu'. Devuelve (device, es_gpu, descripción)."""
    try:
        import torch
    except ImportError:
        return "cpu", False, "CPU (PyTorch no instalado)"
    pref = str(preference or "auto").lower()
    cuda_ok = bool(torch.cuda.is_available())
    if pref.startswith("cuda") and not cuda_ok:
        log.warning("Se pidió %s pero CUDA no está disponible: se usa CPU.", preference)
        pref = "cpu"
    if pref == "auto":
        pref = "cuda:0" if cuda_ok else "cpu"
    if pref.startswith("cuda"):
        idx = int(pref.split(":")[1]) if ":" in pref else 0
        return pref, True, f"GPU {torch.cuda.get_device_name(idx)}"
    return "cpu", False, "CPU" if cuda_ok else "CPU (sin CUDA)"


def resolve_model_path(cfg, is_gpu: bool) -> Path:
    choice = str(cfg.MODEL_PATH)
    if choice.lower() == "auto":
        candidates = [cfg.MODEL_PATH_GPU, cfg.MODEL_PATH_CPU] if is_gpu else [cfg.MODEL_PATH_CPU, cfg.MODEL_PATH_GPU]
    else:
        candidates = [choice]
    for c in candidates:
        p = cfg.resolve_path(c)
        if p.exists():
            return p
    return cfg.resolve_path(candidates[0])     # no existe: Ultralytics intentará descargarlo


def write_tracker_yaml(cfg, out_dir: Path) -> Path:
    """Copia el YAML oficial del tracker elegido con los umbrales de VIMOP."""
    from ultralytics.utils.checks import check_yaml
    try:
        from ultralytics.utils import YAML
        load_yaml, save_yaml = YAML.load, YAML.save
    except ImportError:                          # versiones antiguas de ultralytics
        from ultralytics.utils import yaml_load as load_yaml, yaml_save as save_yaml
    name = str(cfg.TRACKER).lower().replace(".yaml", "")
    if name not in ("bytetrack", "botsort"):
        log.warning("TRACKER '%s' desconocido; se usa bytetrack", name)
        name = "bytetrack"
    params = dict(load_yaml(check_yaml(f"{name}.yaml")))
    params.update(track_high_thresh=float(cfg.CONFIDENCE), track_low_thresh=float(cfg.TRACK_LOW_CONFIDENCE),
                  new_track_thresh=float(cfg.CONFIDENCE), track_buffer=int(cfg.TRACK_BUFFER))
    if name == "botsort":
        params["with_reid"] = False
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"vimop_{name}.yaml"
    save_yaml(path, params)
    return path


class YoloDetector:
    """Detector de motos y personas con tracking integrado (model.track)."""

    def __init__(self, cfg, device: str, is_gpu: bool):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelNotFoundError("Ultralytics no está instalado: pip install -r requirements.txt") from exc
        self.cfg = cfg
        self.device = device
        self.half = bool(cfg.HALF_PRECISION) and is_gpu
        path = resolve_model_path(cfg, is_gpu)
        if not path.exists():
            if not (path.name.startswith("yolo") and path.suffix == ".pt"):
                raise ModelNotFoundError(f"No se encontró el modelo de detección:\n{path}")
            log.info("Descargando %s (requiere internet)...", path.name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.model = YOLO(str(path))
        except Exception as exc:
            raise ModelNotFoundError(
                f"No se pudo cargar/descargar el modelo '{path.name}'.\n"
                f"Si no hay internet, copie el archivo .pt en la carpeta models/.\nDetalle: {exc}") from exc
        self.model_path = path
        self.is_segmentation = getattr(self.model, "task", "") == "segment"
        names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        by_name = {str(v).lower(): int(k) for k, v in names.items()}
        self.names = names
        self.moto_ids = {by_name[n.lower()] for n in cfg.MOTORCYCLE_CLASSES if n.lower() in by_name}
        self.person_ids = {by_name[n.lower()] for n in cfg.PERSON_CLASSES if n.lower() in by_name}
        if not self.moto_ids:
            raise ModelNotFoundError(f"El modelo {path.name} no tiene la clase 'motorcycle'.")
        self.classes = sorted(self.moto_ids | (self.person_ids if cfg.DETECT_PERSONS else set()))
        self.tracker_yaml = write_tracker_yaml(cfg, cfg.resolve_path("config"))
        log.info("Detector listo: %s en %s · tracker %s", path.name, device, self.tracker_yaml.name)

    @property
    def description(self) -> str:
        return f"{self.model_path.name} ({'seg' if self.is_segmentation else 'det'})"

    def track(self, frame: np.ndarray) -> DetectionBatch:
        cfg = self.cfg
        # Confianza baja a propósito: ByteTrack usa detecciones débiles solo para
        # mantener tracks existentes; crea tracks nuevos con CONFIDENCE.
        conf = min(float(cfg.TRACK_LOW_CONFIDENCE), float(cfg.CONFIDENCE))
        res = self.model.track(frame, persist=True, tracker=str(self.tracker_yaml), imgsz=int(cfg.IMG_SIZE),
                               conf=conf, iou=float(cfg.IOU), classes=self.classes, device=self.device,
                               half=self.half, verbose=False)[0]
        return self._parse(res, frame, tracked=True)

    def detect(self, frame: np.ndarray) -> DetectionBatch:
        """Detección de UNA imagen (modo foto): sin tracker, solo detecciones con confianza >= CONFIDENCE."""
        cfg = self.cfg
        res = self.model.predict(frame, imgsz=int(cfg.IMG_SIZE), conf=float(cfg.CONFIDENCE), iou=float(cfg.IOU),
                                 classes=self.classes, device=self.device, half=self.half, verbose=False)[0]
        return self._parse(res, frame, tracked=False)

    def _parse(self, res, frame: np.ndarray, tracked: bool) -> DetectionBatch:
        cfg = self.cfg
        batch = DetectionBatch(inference_ms=float(sum(res.speed.values())) if res.speed else 0.0)
        if res.boxes is None or len(res.boxes) == 0:
            return batch
        h0, w0 = frame.shape[:2]
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [None] * len(xyxy)
        mask_data = res.masks.data.cpu().numpy() if (self.is_segmentation and res.masks is not None) else None

        for i in range(len(xyxy)):
            cls_id = int(clss[i])
            is_moto = cls_id in self.moto_ids
            if not is_moto and float(confs[i]) < float(cfg.CONFIDENCE):
                continue
            x1, y1, x2, y2 = xyxy[i]
            x1, y1 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
            x2, y2 = int(min(w0, np.ceil(x2))), int(min(h0, np.ceil(y2)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            mask = self._mask_to_bbox(mask_data[i], (x1, y1, x2, y2), (h0, w0)) if mask_data is not None else None
            tid = None if ids[i] is None else int(ids[i])
            det = Detection("motorcycle" if is_moto else "person", str(self.names[cls_id]), float(confs[i]),
                            (x1, y1, x2, y2), mask, tid)
            if is_moto:
                if tid is not None or not tracked:  # con tracker: solo motos con track confirmado
                    batch.motorcycles.append(det)
            else:
                batch.persons.append(det)
        return batch

    @staticmethod
    def _mask_to_bbox(mask_in: np.ndarray, bbox, orig_shape) -> np.ndarray | None:
        """Máscara de resolución de inferencia (letterbox) -> región del bbox en el frame original."""
        h0, w0 = orig_shape
        h_in, w_in = mask_in.shape
        gain = min(h_in / h0, w_in / w0)
        pad_x = (w_in - round(w0 * gain)) / 2.0
        pad_y = (h_in - round(h0 * gain)) / 2.0
        x1, y1, x2, y2 = bbox
        mx1 = int(np.clip(np.floor(x1 * gain + pad_x), 0, w_in - 1))
        my1 = int(np.clip(np.floor(y1 * gain + pad_y), 0, h_in - 1))
        mx2 = int(np.clip(np.ceil(x2 * gain + pad_x), mx1 + 1, w_in))
        my2 = int(np.clip(np.ceil(y2 * gain + pad_y), my1 + 1, h_in))
        crop = mask_in[my1:my2, mx1:mx2].astype(np.float32)
        if crop.size == 0:
            return None
        mask = cv2.resize(crop, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR) > 0.5
        return mask if mask.any() else None
