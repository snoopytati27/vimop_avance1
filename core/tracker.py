"""
core/tracker.py — Gestor de identidades: del track_id de Ultralytics a "MOTO #n".

El tracker de bajo nivel (ByteTrack / BoT-SORT) corre dentro de model.track()
(core/detector.py): filtro de Kalman + asociación por IoU (algoritmo húngaro).
Este módulo añade la capa de identidad de VIMOP:

  1. Confirmación: un track_id recibe un número "MOTO #n" solo tras
     TRACK_MIN_HITS frames seguidos (evita IDs por detecciones falsas).
  2. Retención: si una moto deja de verse, su registro se conserva
     ID_RETENTION_SECONDS. Si el tracker la recupera con el MISMO track_id,
     vuelve con su número. Si la recupera con un track_id NUEVO (p. ej. tras una
     oclusión larga), se re-identifica por posición (IoU con su última caja) y
     apariencia (histograma de color H-S). En un parqueadero esto funciona muy bien
     porque las motos estacionadas no se mueven.

Estructuras de datos:
    track_to_moto : dict  track_id -> moto_id           acceso O(1)
    pending_hits  : dict  track_id -> frames vistos      confirmación O(1)
    active        : set   moto_id visibles o recientes   pertenencia O(1)
    lost          : OrderedDict moto_id -> LostMoto      ordenado por momento de pérdida:
                    las entradas vencidas están siempre al inicio -> expiración O(1)
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.detector import Detection
from utils import bbox_iou

log = logging.getLogger("vimop.tracker")


@dataclass
class LostMoto:
    moto_id: int
    bbox: tuple[int, int, int, int]
    appearance: np.ndarray | None
    lost_time: float


@dataclass
class TrackedMoto:
    moto_id: int
    detection: Detection
    is_new: bool = False
    reidentified: bool = False


@dataclass
class TrackingResult:
    tracked: list[TrackedMoto] = field(default_factory=list)
    appeared: set[int] = field(default_factory=set)       # visibles ahora y no en el frame anterior
    disappeared: set[int] = field(default_factory=set)    # visibles antes y no ahora
    newly_lost: list[int] = field(default_factory=list)   # superaron LOST_TIMEOUT_SECONDS
    finalized: list[int] = field(default_factory=list)    # superaron ID_RETENTION_SECONDS


def appearance_histogram(frame: np.ndarray, det: Detection) -> np.ndarray | None:
    x1, y1, x2, y2 = det.bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    mask = det.mask_or_box().astype(np.uint8) * 255
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


class IdentityManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.track_to_moto: dict[int, int] = {}
        self.pending_hits: dict[int, int] = {}
        self.active: set[int] = set()
        self.visible_prev: set[int] = set()
        self.lost: OrderedDict[int, LostMoto] = OrderedDict()
        self.last_seen: dict[int, float] = {}
        self.last_bbox: dict[int, tuple[int, int, int, int]] = {}
        self.appearance: dict[int, np.ndarray] = {}
        self.next_id = 1
        self._frame = 0

    def update(self, detections: list[Detection], frame: np.ndarray, t: float) -> TrackingResult:
        self._frame += 1
        res = TrackingResult()
        seen_tids: set[int] = set()
        visible: set[int] = set()

        for det in detections:
            tid = det.track_id
            if tid is None:
                continue
            seen_tids.add(tid)
            moto_id = self.track_to_moto.get(tid)
            is_new = reid = False
            if moto_id is None:
                hits = self.pending_hits.get(tid, 0) + 1
                self.pending_hits[tid] = hits
                if hits < int(self.cfg.TRACK_MIN_HITS):
                    continue
                del self.pending_hits[tid]
                moto_id = self._reidentify(det, frame, exclude=visible)
                if moto_id is not None:
                    reid = True
                    log.info("track %d re-identificado como MOTO #%d", tid, moto_id)
                else:
                    moto_id, self.next_id, is_new = self.next_id, self.next_id + 1, True
                self.track_to_moto[tid] = moto_id
            if moto_id in visible:
                continue                                  # dos tracks para el mismo objeto
            if moto_id in self.lost:                      # el tracker la recuperó con su track_id
                del self.lost[moto_id]
                reid = True
            visible.add(moto_id)
            self.active.add(moto_id)
            self.last_seen[moto_id] = t
            self.last_bbox[moto_id] = det.bbox
            if moto_id not in self.appearance or self._frame % 10 == moto_id % 10:
                hist = appearance_histogram(frame, det)
                if hist is not None:
                    self.appearance[moto_id] = hist
            res.tracked.append(TrackedMoto(moto_id, det, is_new, reid))

        for tid in list(self.pending_hits):
            if tid not in seen_tids:
                del self.pending_hits[tid]

        # Diferencias de CONJUNTOS entre frames
        res.appeared = visible - self.visible_prev
        res.disappeared = self.visible_prev - visible
        self.visible_prev = visible

        timeout = float(self.cfg.LOST_TIMEOUT_SECONDS)
        for moto_id in list(self.active - visible):
            if t - self.last_seen.get(moto_id, t) > timeout:
                self.active.discard(moto_id)
                self.lost[moto_id] = LostMoto(moto_id, self.last_bbox[moto_id],
                                              self.appearance.get(moto_id), self.last_seen[moto_id])
                res.newly_lost.append(moto_id)

        retention = float(self.cfg.ID_RETENTION_SECONDS)
        while self.lost:
            moto_id, entry = next(iter(self.lost.items()))
            if t - entry.lost_time <= retention:
                break
            self.lost.popitem(last=False)
            self._forget(moto_id)
            res.finalized.append(moto_id)
        return res

    def finalize_all(self) -> list[int]:
        ids = sorted(self.active | set(self.lost))
        self.active.clear()
        self.lost.clear()
        return ids

    def _reidentify(self, det: Detection, frame: np.ndarray, exclude: set[int]) -> int | None:
        if not self.lost:
            return None
        hist = appearance_histogram(frame, det)
        best_id, best_score = None, 0.0
        for moto_id, entry in self.lost.items():
            if moto_id in exclude:
                continue
            iou = bbox_iou(det.bbox, entry.bbox)
            if iou < float(self.cfg.REID_IOU):
                continue
            sim = 1.0
            if hist is not None and entry.appearance is not None:
                sim = float(cv2.compareHist(hist, entry.appearance, cv2.HISTCMP_CORREL))
                if sim < float(self.cfg.REID_APPEARANCE_MIN):
                    continue                    # misma zona, otra moto
            score = iou + 0.5 * sim
            if score > best_score:
                best_id, best_score = moto_id, score
        if best_id is not None:
            del self.lost[best_id]
        return best_id

    def _forget(self, moto_id: int) -> None:
        self.last_seen.pop(moto_id, None)
        self.last_bbox.pop(moto_id, None)
        self.appearance.pop(moto_id, None)
        for tid in [k for k, v in self.track_to_moto.items() if v == moto_id]:
            del self.track_to_moto[tid]
