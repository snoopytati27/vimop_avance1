"""
analysis/state_machine.py — Estados de movimiento de una moto.

    ENTRANDO ──> MOVIÉNDOSE ──> ESTACIONANDO ──> ESTACIONADA
                     ^                                │
                     └──────── (vuelve a moverse) ────┘

- ENTRANDO:     recién aparece; todavía no hay historial para medir velocidad.
- MOVIÉNDOSE:   la velocidad del centroide supera el umbral alto. No se evalúa el parqueo
                (bbox gris, sin score).
- ESTACIONANDO: casi quieta (velocidad bajo el umbral bajo) dentro de un puesto.
                El score se calcula como provisional, sin registrar eventos.
- ESTACIONADA:  quieta durante stable_seconds -> score definitivo, eventos y alertas.
                Si está quieta FUERA de todo puesto se exige el doble de tiempo
                (una moto esperando en la vía de circulación no está "parqueada").

La velocidad es la del centro de la proyección R, dividida por la diagonal del bbox
(diagonales por segundo: no depende de lo lejos que esté la moto), medida sobre una
ventana de `window` segundos. Hay histéresis (umbral alto para empezar a moverse,
bajo para detenerse) para que el estado no parpadee por el ruido del bbox.
Las posiciones se guardan en un deque(maxlen=N): ventana deslizante O(1).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

ST_ENTERING = "ENTRANDO"
ST_MOVING = "MOVIÉNDOSE"
ST_PARKING = "ESTACIONANDO"
ST_PARKED = "ESTACIONADA"
EVALUATED_STATES = (ST_PARKING, ST_PARKED)


@dataclass
class MotionParams:
    window: float = 0.8            # s
    speed_high: float = 0.30       # diagonales/s para pasar a MOVIÉNDOSE
    speed_low: float = 0.12        # diagonales/s para considerarla quieta
    stable_seconds: float = 2.0
    outside_factor: float = 2.0


class MotionStateMachine:
    def __init__(self, params: MotionParams, history: int = 90):
        self.p = params
        self.state = ST_ENTERING
        self.still_since: float | None = None
        self.positions: deque[tuple[float, float, float]] = deque(maxlen=history)   # (t, x, y)
        self.speed: float | None = None

    def update(self, t: float, pos: tuple[float, float], in_space: bool, scale: float = 1.0) -> tuple[str, str]:
        """
        Registra una posición y devuelve (estado_anterior, estado_nuevo).
        `scale` divide la velocidad (la diagonal del bbox).
        """
        self.positions.append((t, float(pos[0]), float(pos[1])))
        prev = self.state
        ref = next((pt for pt in self.positions if t - pt[0] <= self.p.window), None)
        if ref is None or (t - ref[0]) < 0.5 * self.p.window:
            return prev, prev                                   # historial insuficiente
        self.speed = float(np.hypot(pos[0] - ref[1], pos[1] - ref[2])) / max(t - ref[0], 1e-6) / max(scale, 1e-9)

        moving = self.speed > self.p.speed_high or (prev == ST_MOVING and self.speed > self.p.speed_low)
        if moving:
            self.state = ST_MOVING
            self.still_since = None
        else:
            if self.still_since is None:
                self.still_since = t
            needed = self.p.stable_seconds * (1.0 if in_space else self.p.outside_factor)
            if prev == ST_PARKED or t - self.still_since >= needed:
                self.state = ST_PARKED
            else:
                self.state = ST_PARKING
        return prev, self.state
