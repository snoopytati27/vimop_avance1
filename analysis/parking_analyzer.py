"""
analysis/parking_analyzer.py — Análisis del parqueo con la LÍNEA DE EJES de la moto.

Modelo: cada moto es el segmento L = R–F entre el punto de contacto de la rueda trasera (R) y el de la
delantera (F) con el piso; S es el polígono de su puesto, de eje mayor θ_S (calculado una vez en la
calibración), largo L_S y ancho W_S. El bbox se sigue usando para detección y tracking, no aquí.
R y F los entrega WheelLocator (modelo de puntos clave o respaldo por máscara).

Perspectiva: si la calibración trae 4 puntos del piso (ground_quad), R, F y los puestos se pasan a la
vista cenital con una homografía antes de medir ángulos y distancias. Sin ella se mide en la imagen y
los ángulos son aproximados (aviso en `warnings`). La homografía se calibra con UN puesto; lejos de él
puede dispararse por la distorsión de la lente (RuntimeSpace.ground_reliable, en analysis/parking_spaces.py),
así que cada puesto usa su propio plano: el piso si es fiable, si no la imagen (mismo aviso que sin piso).

Fórmulas (Sj = los demás puestos):
    Puesto asignado  argmax_i  long(L ∩ Si) / long(L)   (si ≥ MIN_ASSIGN = 30 %; si no, FUERA)
    Orientación  O   θ_M = atan2(F − R);  d = |θ_M − θ_S| mod 180;  err = min(d, 180 − d)
                     O = 100 · max(0, 1 − err / ERR_MAX)
    Ubicación    U   long(L ∩ S) / long(L) · 100          (shapely.LineString.intersection)
    Centrado     C   M = (R + F) / 2;  (dl, dt) = componentes de M − centro(S) a lo largo del eje y
                     perpendicular;  d_n = sqrt((dl / (L_S/2))² + (dt / (W_S/2))²)
                     C = 100 · max(0, 1 − d_n)
    Invasión     I   ratio = max_j long(L ∩ Sj) / long(L);  I = 100 · (1 − min(1, ratio / 0.30))
                     nivel: 0–2 % ninguna · 2–10 % pequeña · > 10 % considerable
    Score        P   0.35·O + 0.35·U + 0.20·C + 0.10·I  ∈ [0, 100]

Caso degenerado (moto de frente o de espaldas): si |L| < MIN_AXIS_LEN_RATIO · alto del bbox, o no hay ni
modelo de ruedas ni máscara (solo bbox), la orientación es NO CONFIABLE: O = None, no se inventa ningún
ángulo y el score se calcula con los pesos restantes renormalizados.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString

from analysis.geometry import angular_error_deg, segment_axis_angle_deg, transform_points
from analysis.parking_spaces import RuntimeLayout, RuntimeSpace
from analysis.wheels import SOURCE_LABELS, SRC_BBOX, WheelLocator
from core.detector import Detection
from utils import classify_score

log = logging.getLogger("vimop.analyzer")

INV_NONE = "NINGUNA"
INV_SMALL = "PEQUEÑA"
INV_BIG = "CONSIDERABLE"

WARN_UNRELIABLE = "orientación no confiable"
WARN_APPROX = "ángulos aproximados (sin homografía del piso)"


@dataclass
class ParkingAnalysis:
    space_id: str | None
    overlaps: dict[str, float] = field(default_factory=dict)    # puesto -> long(L ∩ S) / long(L)
    orientation: float | None = 0.0   # O; None si la orientación no es confiable
    location: float = 0.0             # U
    centering: float = 0.0            # C
    invasion: float = 100.0           # I (100 = no invade)
    invasion_ratio: float = 0.0
    invasion_level: str = INV_NONE
    invaded_spaces: list[str] = field(default_factory=list)
    angle_error: float | None = None  # err en grados (None si no es confiable o está FUERA)
    orientation_reliable: bool = True
    wheel_source: str = "mask"        # pose | mask | bbox
    wheel_confidence: float = 0.0
    rear: tuple[float, float] = (0.0, 0.0)    # R en la imagen
    front: tuple[float, float] = (0.0, 0.0)   # F en la imagen
    center_img: tuple[float, float] = (0.0, 0.0)   # M = (R + F) / 2 en la imagen
    axis_length: float = 0.0          # |L| en la imagen (px)
    ground_plane: bool = False        # True = medido en vista cenital (homografía)
    warnings: list[str] = field(default_factory=list)
    score: float = 0.0
    classification: str = "PÉSIMO"

    @property
    def outside(self) -> bool:
        return self.space_id is None

    @property
    def wheels_label(self) -> str:
        return "Ruedas: " + SOURCE_LABELS.get(self.wheel_source, self.wheel_source)


class ParkingAnalyzer:
    def __init__(self, cfg, locator: WheelLocator | None = None):
        self.cfg = cfg
        self.w_o, self.w_u, self.w_c, self.w_i = cfg.weights()
        self.locator = locator or WheelLocator(cfg)

    def score(self, o: float | None, u: float, c: float, i: float) -> float:
        """Score 0–100. Con o=None (orientación no confiable) los otros pesos se renormalizan."""
        if o is None:
            total = self.w_u + self.w_c + self.w_i
            if total <= 0:
                return 0.0
            raw = (u * self.w_u + c * self.w_c + i * self.w_i) / total
        else:
            raw = o * self.w_o + u * self.w_u + c * self.w_c + i * self.w_i
        # se redondea para que el ruido de coma flotante (10.000000000000002) no cambie una categoría
        return round(float(np.clip(raw, 0.0, 100.0)), 4)

    def analyze(self, det: Detection, layout: RuntimeLayout, frame: np.ndarray | None = None,
                wheels: dict | None = None) -> ParkingAnalysis | None:
        cfg = self.cfg
        wheels = wheels or self.locator.locate(frame, det)
        r_img = np.asarray(wheels["rear"], dtype=np.float64)
        f_img = np.asarray(wheels["front"], dtype=np.float64)
        axis_len_img = float(np.linalg.norm(f_img - r_img))
        if axis_len_img < 1e-6:
            return None
        m_img = (r_img + f_img) / 2.0
        a = ParkingAnalysis(None, wheel_source=wheels["source"], wheel_confidence=float(wheels.get("confidence", 0.0)),
                            rear=(float(r_img[0]), float(r_img[1])), front=(float(f_img[0]), float(f_img[1])),
                            center_img=(float(m_img[0]), float(m_img[1])), axis_length=axis_len_img)

        # ---- ¿la orientación es confiable? (moto de frente/espaldas, o sin ruedas ni máscara) ----
        bbox_h = max(float(det.bbox[3] - det.bbox[1]), 1.0)
        a.orientation_reliable = (wheels["source"] != SRC_BBOX
                                  and axis_len_img >= float(cfg.MIN_AXIS_LEN_RATIO) * bbox_h)
        if not a.orientation_reliable:
            a.warnings.append(WARN_UNRELIABLE)
            a.orientation = None

        # ---- la línea R-F, en la imagen y (si hay piso) en la vista cenital ----
        line_img = LineString([tuple(r_img), tuple(f_img)])
        r_gnd, f_gnd = transform_points([r_img, f_img], layout.homography)
        line_gnd = LineString([tuple(r_gnd), tuple(f_gnd)])
        length_gnd = line_gnd.length
        if length_gnd < 1e-9:
            return None
        m_gnd = (r_gnd + f_gnd) / 2.0

        # ---- fracción de la longitud de L dentro de cada puesto ----
        # Cada puesto usa SU propio plano: el piso si es fiable (RuntimeSpace.ground_reliable), si no la imagen,
        # igual que si no hubiera homografía. Los dos son comparables porque la fracción no tiene unidades.
        for sp in layout.spaces:
            r, f, line, length, shape, geom = ((r_gnd, f_gnd, line_gnd, length_gnd, sp.work_shape, sp.work_geom)
                                               if sp.ground_reliable else
                                               (r_img, f_img, line_img, axis_len_img, sp.shape, sp.geom))
            minx, miny, maxx, maxy = geom.bounds
            if max(r[0], f[0]) < minx or min(r[0], f[0]) > maxx or max(r[1], f[1]) < miny or min(r[1], f[1]) > maxy:
                continue
            inter = line.intersection(shape).length
            if inter > 0:
                a.overlaps[sp.space_id] = min(1.0, inter / length)

        # ---- puesto asignado ----
        own: RuntimeSpace | None = None
        if a.overlaps:
            best = max(a.overlaps, key=a.overlaps.get)
            if a.overlaps[best] >= float(cfg.MIN_ASSIGN):
                own = layout.by_id[best]
        a.space_id = own.space_id if own else None
        # FUERA de puestos: no hay un puesto propio que decida el plano; se informa según si hay piso calibrado.
        a.ground_plane = bool(own.ground_reliable) if own else layout.homography is not None
        if not a.ground_plane:
            a.warnings.append(WARN_APPROX)

        # ---- invasión: máximo sobre los OTROS puestos ----
        others = [(ratio, sid) for sid, ratio in a.overlaps.items() if own is None or sid != own.space_id]
        a.invasion_ratio = max((ratio for ratio, _ in others), default=0.0)
        a.invaded_spaces = [sid for ratio, sid in sorted(others, reverse=True) if ratio >= float(cfg.INVASION_NONE)]
        a.invasion = 100.0 * (1.0 - min(1.0, a.invasion_ratio / max(float(cfg.INVASION_MAX), 1e-9)))
        if a.invasion_ratio < float(cfg.INVASION_NONE):
            a.invasion_level = INV_NONE
        elif a.invasion_ratio <= float(cfg.INVASION_SMALL):
            a.invasion_level = INV_SMALL
        else:
            a.invasion_level = INV_BIG

        if own is None:                                   # FUERA de puestos: O = U = C = 0
            a.orientation = 0.0 if a.orientation_reliable else None
            return self._finish(a)

        # ---- del puesto asignado en adelante, todo en SU plano (piso o imagen) ----
        r, f, m, g = (r_gnd, f_gnd, m_gnd, own.work_geom) if own.ground_reliable else (r_img, f_img, m_img, own.geom)
        # ---- ubicación ----
        a.location = a.overlaps[own.space_id] * 100.0
        # ---- centrado: M − centro(S) en componentes longitudinal (dl) y transversal (dt) ----
        rad = math.radians(g.angle)
        ux, uy = math.cos(rad), math.sin(rad)
        dx, dy = float(m[0]) - g.center[0], float(m[1]) - g.center[1]
        dl, dt = dx * ux + dy * uy, -dx * uy + dy * ux
        d_n = math.hypot(dl / max(g.length / 2.0, 1e-9), dt / max(g.breadth / 2.0, 1e-9))
        a.centering = 100.0 * max(0.0, 1.0 - d_n)
        # ---- orientación ----
        if a.orientation_reliable:
            a.angle_error = angular_error_deg(segment_axis_angle_deg(r, f), g.angle)
            a.orientation = 100.0 * max(0.0, 1.0 - a.angle_error / max(float(cfg.ERR_MAX), 1e-9))
        return self._finish(a)

    def _finish(self, a: ParkingAnalysis) -> ParkingAnalysis:
        a.score = self.score(a.orientation, a.location, a.centering, a.invasion)
        a.classification = classify_score(a.score)
        return a
