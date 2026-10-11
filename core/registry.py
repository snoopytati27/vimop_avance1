"""
core/registry.py — Registro de motocicletas: fichas, historiales, votaciones y eventos.

ESTRUCTURAS DE DATOS (justificación)
    motos : dict[int, MotoRecord]         TABLA HASH. Cada frame se busca la ficha de cada
                                          moto por su ID: O(1) promedio (una lista sería O(n)).
    active_ids / visible_ids : set[int]   CONJUNTOS. Pertenencia O(1) y diferencias entre
                                          frames: ocultas = activas − visibles.
    plate_history, score_history, position_history
        : deque(maxlen=N)                 COLA DOBLE ACOTADA = ventana deslizante. Insertar la
                                          lectura nueva descarta la más vieja en O(1)
                                          (list.pop(0) sería O(n)). Así un frame malo no destruye
                                          los datos: se decide por votación / promedio.
    Counter                               tabla hash de conteo para las votaciones, O(k).

Reglas de estabilidad
    - Placa: votación ponderada por confianza. Se muestra solo si hay al menos
      PLATE_MIN_VOTES lecturas iguales, la ganadora tiene ≥ PLATE_MIN_CONSENSUS del peso
      total y su confianza media ≥ PLATE_MIN_CONFIDENCE. Si no -> "No detectada".
    - Score: promedio de los últimos HISTORY_LENGTH análisis; la categoría usa histéresis.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from analysis.parking_analyzer import ParkingAnalysis
from analysis.state_machine import (EVALUATED_STATES, ST_MOVING, ST_PARKED, MotionParams, MotionStateMachine)
from utils import NO_PLATE, classify_with_hysteresis

EV_ENTRY = "ENTRADA"
EV_CLASS = "CAMBIO_ESTADO"
EV_SPACE = "CAMBIO_PUESTO"
EV_EXIT = "SALIDA"
EV_PLATE = "PLACA_IDENTIFICADA"
EV_ALERT = "ALERTA"
EV_GONE = "FUERA_DE_VISTA"


@dataclass
class ParkingEvent:
    timestamp: datetime
    moto_id: int
    event_type: str
    space: str | None
    score: float | None
    classification: str | None
    plate: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {"fecha": self.timestamp.strftime("%Y-%m-%d"), "hora": self.timestamp.strftime("%H:%M:%S"),
                "id": self.moto_id, "evento": self.event_type, "puesto": self.space or "",
                "score": "" if self.score is None else round(self.score, 1),
                "estado": self.classification or "", "placa": self.plate, "detalle": self.detail}


def score_row(a: ParkingAnalysis) -> tuple[float, float, float, float, float]:
    """(O, U, C, I, P) para el historial; O = NaN cuando la orientación no es confiable."""
    o = float("nan") if a.orientation is None else float(a.orientation)
    return o, a.location, a.centering, a.invasion, a.score


@dataclass
class MotoRecord:
    moto_id: int
    first_seen: datetime
    motion: MotionStateMachine
    hist_len: int = 15
    ident_len: int = 30

    last_seen: datetime | None = None
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0
    visible: bool = True
    analysis: ParkingAnalysis | None = None
    parked_since: datetime | None = None

    plate_history: deque = field(init=False)
    score_history: deque = field(init=False)      # (O, U, C, I, P); O = NaN si la orientación no es confiable
    space_history: deque = field(init=False)

    display_class: str | None = None               # categoría con histéresis
    committed_class: str | None = None             # última registrada en eventos
    committed_space: str | None = None
    pending_class: str | None = None
    pending_since: float = 0.0
    pending_space: str | None = None
    pending_space_since: float = 0.0
    plate_announced: bool = False
    last_ocr_frame: int = -10 ** 9

    def __post_init__(self):
        self.plate_history = deque(maxlen=self.ident_len)
        self.score_history = deque(maxlen=self.hist_len)
        self.space_history = deque(maxlen=self.hist_len)

    @property
    def state(self) -> str:
        return self.motion.state

    @property
    def position_history(self) -> deque:
        return self.motion.positions

    # ---- identificación estable ------------------------------------------------
    def plate_vote(self) -> tuple[str | None, int, float, float]:
        """(placa ganadora, nº lecturas iguales, confianza media, fracción del peso total)."""
        if not self.plate_history:
            return None, 0, 0.0, 0.0
        weight: Counter = Counter()
        count: Counter = Counter()
        for text, conf in self.plate_history:
            weight[text] += conf
            count[text] += 1
        best, w = weight.most_common(1)[0]
        total = sum(weight.values())
        return best, count[best], w / count[best], w / max(total, 1e-9)

    def plate(self, cfg) -> str:
        text, votes, conf, share = self.plate_vote()
        if (text and votes >= int(cfg.PLATE_MIN_VOTES) and share >= float(cfg.PLATE_MIN_CONSENSUS)
                and conf >= float(cfg.PLATE_MIN_CONFIDENCE)):
            return text
        return NO_PLATE

    # ---- puntaje ------------------------------------------------------------------
    def smoothed(self) -> tuple[float, float, float, float, float] | None:
        if not self.score_history:
            return None
        h = np.asarray(self.score_history, dtype=np.float64)        # O es NaN si la orientación no es confiable
        o = float(np.nanmean(h[:, 0])) if not np.isnan(h[:, 0]).all() else float("nan")
        m = h.mean(axis=0)
        return o, float(m[1]), float(m[2]), float(m[3]), float(m[4])

    def smoothed_score(self) -> float | None:
        s = self.smoothed()
        return None if s is None else s[4]

    def space(self) -> str | None:
        if not self.space_history:
            return self.analysis.space_id if self.analysis else None
        return Counter(self.space_history).most_common(1)[0][0]

    @property
    def evaluated(self) -> bool:
        return self.state in EVALUATED_STATES and bool(self.score_history)

    def to_dict(self, cfg) -> dict:
        sm = self.smoothed() if self.evaluated else None
        a = self.analysis
        plate_text, votes, _, _ = self.plate_vote()
        return {
            "id": self.moto_id,
            "placa": self.plate(cfg),
            "placa_lecturas": votes if plate_text else 0,
            "puesto": (self.space() or "FUERA") if sm else "",
            "estado_movimiento": self.state,
            "visible": self.visible,
            "orientacion": round(sm[0], 1) if (sm and not math.isnan(sm[0])) else None,
            "ubicacion": round(sm[1], 1) if sm else None,
            "centrado": round(sm[2], 1) if sm else None,
            "invasion": round(sm[3], 1) if sm else None,
            "nivel_invasion": a.invasion_level if (a and sm) else "",
            "puestos_invadidos": list(a.invaded_spaces) if (a and sm) else [],
            "error_angular": round(a.angle_error, 1) if (a and sm and a.angle_error is not None) else None,
            "orientacion_confiable": bool(a.orientation_reliable) if (a and sm) else None,
            "ruedas_origen": a.wheel_source if (a and sm) else "",
            "avisos": list(a.warnings) if (a and sm) else [],
            "puntaje": round(sm[4], 1) if sm else None,
            "clasificacion": self.display_class if sm else None,
            "bbox": list(self.bbox),
            "primera_vez": self.first_seen.strftime("%Y-%m-%d %H:%M:%S"),
            "ultima_vez": (self.last_seen or self.first_seen).strftime("%Y-%m-%d %H:%M:%S"),
            "estacionada_desde": self.parked_since.strftime("%H:%M:%S") if self.parked_since else "",
        }


class MotoRegistry:
    def __init__(self, cfg):
        self.cfg = cfg
        self.motos: dict[int, MotoRecord] = {}
        self.active_ids: set[int] = set()
        self.visible_ids: set[int] = set()
        self.finalized_ids: set[int] = set()

    def motion_params(self) -> MotionParams:
        cfg = self.cfg
        return MotionParams(float(cfg.MOTION_WINDOW_SECONDS), float(cfg.MOVING_SPEED_HIGH),
                            float(cfg.MOVING_SPEED_LOW), float(cfg.PARKED_STABLE_SECONDS))

    def get(self, moto_id: int) -> MotoRecord | None:
        return self.motos.get(moto_id)

    def active(self) -> list[MotoRecord]:
        return [self.motos[i] for i in sorted(self.active_ids)]

    def begin_frame(self) -> None:
        self.visible_ids = set()

    def observe(self, moto_id: int, bbox, conf: float, now: datetime) -> MotoRecord:
        rec = self.motos.get(moto_id)
        if rec is None:
            cfg = self.cfg
            rec = MotoRecord(moto_id, now, MotionStateMachine(self.motion_params(),
                                                              int(cfg.POSITION_HISTORY_LENGTH)),
                             int(cfg.HISTORY_LENGTH), int(cfg.IDENT_HISTORY_LENGTH))
            self.motos[moto_id] = rec
        rec.bbox = tuple(int(v) for v in bbox)
        rec.confidence, rec.last_seen, rec.visible = conf, now, True
        self.finalized_ids.discard(moto_id)
        self.active_ids.add(moto_id)
        self.visible_ids.add(moto_id)
        return rec

    def mark_hidden(self, moto_ids) -> None:
        for moto_id in moto_ids:
            if moto_id in self.motos:
                self.motos[moto_id].visible = False

    def finalize(self, moto_id: int) -> MotoRecord | None:
        self.active_ids.discard(moto_id)
        self.visible_ids.discard(moto_id)
        self.finalized_ids.add(moto_id)
        rec = self.motos.get(moto_id)
        if rec:
            rec.visible = False
        return rec

    # ------------------------------------------------------------------
    def update_motion(self, rec: MotoRecord, t: float, pos, in_space: bool, scale: float,
                      now: datetime) -> list[ParkingEvent]:
        prev, new = rec.motion.update(t, pos, in_space, scale)
        events: list[ParkingEvent] = []
        if new == prev:
            return events
        if new == ST_PARKED:
            rec.parked_since = now
        if prev == ST_PARKED and new == ST_MOVING:
            events.append(ParkingEvent(now, rec.moto_id, EV_EXIT, rec.committed_space, rec.smoothed_score(),
                                       rec.committed_class, rec.plate(self.cfg), "La moto empezó a moverse"))
            rec.parked_since = None
        if new == ST_MOVING:                       # un parqueo nuevo empieza de cero
            rec.score_history.clear()
            rec.space_history.clear()
            rec.display_class = rec.committed_class = rec.committed_space = None
            rec.pending_class = rec.pending_space = None
        return events

    def apply_analysis(self, rec: MotoRecord, analysis: ParkingAnalysis | None, t: float,
                       now: datetime) -> list[ParkingEvent]:
        cfg = self.cfg
        rec.analysis = analysis
        if analysis is None or rec.state not in EVALUATED_STATES:
            return []
        rec.score_history.append(score_row(analysis))
        rec.space_history.append(analysis.space_id)
        score = rec.smoothed_score()
        rec.display_class = classify_with_hysteresis(score, rec.display_class, float(cfg.CLASS_HYSTERESIS))
        if rec.state != ST_PARKED:
            return []

        cls, space, plate = rec.display_class, rec.space(), rec.plate(cfg)
        if rec.committed_class is None:
            rec.committed_class, rec.committed_space = cls, space
            return [ParkingEvent(now, rec.moto_id, EV_ENTRY, space, score, cls, plate,
                                 "Moto estacionada" if space else "Moto detenida FUERA de los puestos")]
        events = []
        if cls != rec.committed_class:
            if rec.pending_class != cls:
                rec.pending_class, rec.pending_since = cls, t
            elif t - rec.pending_since >= float(cfg.CLASS_CHANGE_MIN_SECONDS):
                events.append(ParkingEvent(now, rec.moto_id, EV_CLASS, space, score, cls, plate,
                                           f"{rec.committed_class} -> {cls}"))
                rec.committed_class, rec.pending_class = cls, None
        else:
            rec.pending_class = None
        # cambio de puesto: también debe sostenerse (una moto en el borde no genera eventos repetidos)
        if space != rec.committed_space:
            if rec.pending_space != space:
                rec.pending_space, rec.pending_space_since = space, t
            elif t - rec.pending_space_since >= float(cfg.CLASS_CHANGE_MIN_SECONDS):
                events.append(ParkingEvent(now, rec.moto_id, EV_SPACE, space, score, cls, plate,
                                           f"{rec.committed_space or 'FUERA'} -> {space or 'FUERA'}"))
                rec.committed_space, rec.pending_space = space, None
        else:
            rec.pending_space = None
        return events

    def add_plate(self, rec: MotoRecord, text: str | None, conf: float, method: str,
                  now: datetime) -> list[ParkingEvent]:
        if not text:
            return []
        rec.plate_history.append((text, conf))
        plate = rec.plate(self.cfg)
        if plate != NO_PLATE and not rec.plate_announced:
            rec.plate_announced = True
            return [ParkingEvent(now, rec.moto_id, EV_PLATE, rec.space(), rec.smoothed_score(),
                                 rec.display_class, plate, f"Placa confirmada ({method})")]
        return []
