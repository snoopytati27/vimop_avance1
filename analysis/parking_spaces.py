"""
analysis/parking_spaces.py — Puestos de parqueo (P01, P02, ...) y su persistencia.

Cada puesto es un POLÍGONO en coordenadas de la imagen de ESTA cámara.

config/parking_spaces.json
{
  "version": 4,
  "reference_resolution": [1322, 604],      # imagen sobre la que se calibró
  "units": "px",
  "spaces": {
    "P01": {
      "points": [[x, y], ...],     # vértices (px)
      "axis_swap": false,          # el usuario rotó el eje del puesto (afecta a la orientación y al dibujo)
      "center": [x, y],            # centroide del polígono
      "angle": 63.4,               # eje largo del polígono (grados, 0–180)
      "width": 120.0,              # ancho del bbox del puesto (px)
      "height": 210.0              # alto del bbox del puesto (px)
    }
  }
}
center, angle, width y height se calculan UNA vez al guardar/cargar: los puestos son estáticos.

Opcional, para medir sin perspectiva (homografía a vista cenital):
  "ground_quad": {"image_points": [[x, y] x4],   # 4 esquinas de un rectángulo o paralelogramo real del piso
                  "size": [ancho, largo],        # lado 1→2 y lado 2→3 (cualquier unidad)
                  "angle": 90.0}                 # ángulo interior en la esquina 1 (grados); 90 = rectángulo
Con ella, ángulos y distancias se miden en el plano del piso; sin ella, en la imagen (aproximado).
Un puesto en "romboide" (paralelogramo) sirve como referencia si se conoce su ángulo: parallelogram_angle()
lo calcula a partir de los lados y de las diagonales medidas.
"reference_thumbnail" guarda una miniatura en grises (96x54) de la imagen de calibración para
comprobar, en modo foto, que una foto tiene el mismo PUNTO DE VISTA que la calibración.

Compatibilidad foto / calibración (check_compatibility):
    misma resolución                      -> "ok"
    otra resolución, misma proporción     -> "scaled"   (los polígonos se escalan)
    otra proporción                       -> "mismatch" (no se pueden aplicar los puestos)
    punto de vista distinto (miniatura)   -> "mismatch"
"""
from __future__ import annotations

import base64
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from analysis.geometry import aspect_ratio, min_area_rect_axis, space_polygon, transform_points

log = logging.getLogger("vimop.spaces")

THUMB_SIZE = (96, 54)            # miniatura de la imagen de calibración (ancho, alto)
ASPECT_TOLERANCE = 0.02          # 2 % de diferencia de proporción se considera la misma


def make_thumbnail(image_bgr: np.ndarray) -> str:
    """Miniatura en grises, suavizada, codificada en base64 (para guardar en el JSON)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    small = cv2.resize(gray, THUMB_SIZE, interpolation=cv2.INTER_AREA)
    return base64.b64encode(small.astype(np.uint8).tobytes()).decode("ascii")


def viewpoint_similarity(thumbnail_b64: str, image_bgr: np.ndarray) -> float | None:
    """Correlación normalizada (−1..1) entre la miniatura de calibración y la foto; None si no hay miniatura."""
    try:
        ref = np.frombuffer(base64.b64decode(thumbnail_b64), dtype=np.uint8).reshape(THUMB_SIZE[1], THUMB_SIZE[0])
    except (ValueError, TypeError):
        return None
    cur = np.frombuffer(base64.b64decode(make_thumbnail(image_bgr)), dtype=np.uint8).reshape(THUMB_SIZE[1],
                                                                                              THUMB_SIZE[0])
    a = cv2.GaussianBlur(ref, (5, 5), 0).astype(np.float32)
    b = cv2.GaussianBlur(cur, (5, 5), 0).astype(np.float32)
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0][0])


@dataclass
class Compatibility:
    status: str                  # "ok" | "scaled" | "mismatch" | "none"
    message: str
    similarity: float | None = None

    @property
    def usable(self) -> bool:
        return self.status in ("ok", "scaled")


def ground_quad_error(points) -> str | None:
    """Mensaje si los 4 puntos del piso no sirven; None si son válidos (cuadrilátero convexo, sin cruzar)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) != 4:
        return "Se necesitan exactamente 4 puntos."
    edges = [pts[(i + 1) % 4] - pts[i] for i in range(4)]
    cross = [float(edges[i][0] * edges[(i + 1) % 4][1] - edges[i][1] * edges[(i + 1) % 4][0]) for i in range(4)]
    if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
        return ("Los 4 puntos deben formar un cuadrilátero convexo, en orden alrededor del rectángulo "
                "(sin cruzar los lados ni quedar alineados).")
    if space_polygon(pts).area < 400.0:
        return "El cuadrilátero es demasiado pequeño: use un rectángulo más grande del piso."
    return None


ANGLE_MIN, ANGLE_MAX = 10.0, 170.0          # fuera de este rango el paralelogramo es casi una línea
COS_TOLERANCE = 0.02                        # error de medición admitido al calcular el ángulo con el coseno

GROUND_AREA_TOLERANCE = 3.0                 # ver ground_reliable() en RuntimeLayout.runtime()


def parallelogram_angle(width: float, length: float, diag_13: float | None = None,
                        diag_24: float | None = None) -> float:
    """Ángulo interior (grados) en la esquina 1 de un paralelogramo con lados 1→2 = width y 2→3 = length.

    diag_13 (esquina 1 a 3) y diag_24 (2 a 4) son opcionales; con las dos se promedian. Sin ninguna: 90°
    (rectángulo). Lanza ValueError si las medidas no forman un paralelogramo posible.
    """
    a, b = float(width), float(length)
    if a <= 0 or b <= 0:
        raise ValueError("los lados deben ser positivos")
    diagonals = [d for d in (diag_13, diag_24) if d]
    if not diagonals:
        return 90.0
    cosines = []
    if diag_13:
        cosines.append((float(diag_13) ** 2 - a * a - b * b) / (2 * a * b))    # d13² = a² + b² + 2ab·cosθ
    if diag_24:
        cosines.append((a * a + b * b - float(diag_24) ** 2) / (2 * a * b))    # d24² = a² + b² − 2ab·cosθ
    for c in cosines:
        if abs(c) > 1.0 + COS_TOLERANCE:
            raise ValueError("las diagonales no son posibles con esos lados (vuelva a medir)")
    cos_t = max(-1.0, min(1.0, sum(cosines) / len(cosines)))
    angle = math.degrees(math.acos(cos_t))
    if not ANGLE_MIN <= angle <= ANGLE_MAX:
        raise ValueError(f"el ángulo resultante ({angle:.0f}°) es casi una línea (vuelva a medir)")
    return angle


def ground_destination(size, angle: float, src_points) -> np.ndarray:
    """Las 4 esquinas del piso en la vista cenital: rectángulo o paralelogramo con el sentido de giro de la imagen."""
    gw, gh = float(size[0]), float(size[1])
    t = math.radians(float(angle))
    p4 = (gh * math.cos(t), gh * math.sin(t))
    dst = np.array([(0.0, 0.0), (gw, 0.0), (gw + p4[0], p4[1]), p4], dtype=np.float64)
    src = np.asarray(src_points, dtype=np.float64).reshape(4, 2)
    turn = float(np.sum(src[:, 0] * np.roll(src[:, 1], -1) - np.roll(src[:, 0], -1) * src[:, 1]))
    if turn < 0:                                     # esquinas marcadas en sentido contrario: no reflejar la vista
        dst[:, 1] *= -1.0
    return dst


def clean_polygon_points(points, tol: float = 3.0) -> list[tuple[float, float]]:
    """
    Quita vértices repetidos: dos seguidos casi en el mismo lugar, o el último casi encima del primero. Cerrar un puesto
    con doble clic dejaba un vértice de más (5 en vez de 4) que impedía, por ejemplo, dividirlo.
    """
    pts = [(float(x), float(y)) for x, y in points]
    out: list[tuple[float, float]] = []
    for q in pts:
        if not out or math.hypot(q[0] - out[-1][0], q[1] - out[-1][1]) > tol:
            out.append(q)
    while len(out) > 3 and math.hypot(out[-1][0] - out[0][0], out[-1][1] - out[0][1]) <= tol:
        out.pop()
    return out if len(out) >= 3 else pts             # si no queda un polígono, se deja como estaba


def split_space_points(points, n: int, axis_angle: float | None = None) -> list[list[tuple[float, float]]]:
    """
    Parte un puesto de 4 esquinas en `n` tramos iguales A LO LARGO de su eje, respetando la perspectiva.

    Las esquinas se toman como la imagen de un rectángulo; con la homografía rectángulo -> puesto se marcan cortes
    igualmente espaciados en el piso real (los tramos lejanos salen más chicos en la imagen, como corresponde).
    `axis_angle` (grados) elige en qué dirección se corta: la de los lados más cercanos a ese ángulo; sin él, la de
    los lados más largos en la imagen. Devuelve `n` listas de 4 esquinas, ordenadas a lo largo del eje.
    """
    pts = clean_polygon_points(points)
    if len(pts) != 4:
        raise ValueError("Solo se pueden dividir puestos de 4 esquinas.")
    if not 2 <= int(n) <= 20:
        raise ValueError("El número de partes debe estar entre 2 y 20.")
    if ground_quad_error(pts):
        raise ValueError("El puesto debe ser un cuadrilátero convexo (sin lados cruzados).")
    arr = np.asarray(pts, dtype=np.float64)

    def edge_angle(a, b):
        return math.degrees(math.atan2(arr[b][1] - arr[a][1], arr[b][0] - arr[a][0])) % 180.0

    def edge_len(a, b):
        return float(np.linalg.norm(arr[b] - arr[a]))

    # Pareja de lados que corre a lo largo del eje: (0-1 y 3-2) o (1-2 y 0-3)
    if axis_angle is not None:
        def gap(a, b):
            d = abs(edge_angle(a, b) - axis_angle) % 180.0
            return min(d, 180.0 - d)
        first = gap(0, 1) + gap(3, 2) <= gap(1, 2) + gap(0, 3)
    else:
        first = edge_len(0, 1) + edge_len(3, 2) >= edge_len(1, 2) + edge_len(0, 3)
    if not first:                                              # girar los vértices: el lado 1-2 pasa a ser el "largo"
        arr = np.roll(arr, -1, axis=0)
    h, _ = cv2.findHomography(np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float64), arr)   # float64: sin redondeo

    def mapped(u: float, v: float) -> tuple[float, float]:
        q = h @ np.array([u, v, 1.0])
        return float(q[0] / q[2]), float(q[1] / q[2])

    cuts = [i / n for i in range(n + 1)]
    return [[mapped(cuts[i], 0.0), mapped(cuts[i + 1], 0.0), mapped(cuts[i + 1], 1.0), mapped(cuts[i], 1.0)]
            for i in range(n)]


def natural_key(space_id: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", space_id)]


@dataclass
class SpaceGeometry:
    center: tuple[float, float]                   # centroide
    angle: float                                  # eje largo (grados)
    width: float                                  # ancho del bbox
    height: float                                 # alto del bbox
    bounds: tuple[float, float, float, float]     # bbox (minx, miny, maxx, maxy)
    axis: tuple[tuple[float, float], tuple[float, float]]   # eje largo, para dibujar
    length: float = 0.0                           # L_S: extensión del puesto A LO LARGO de su eje
    breadth: float = 0.0                          # W_S: extensión del puesto perpendicular a su eje


def space_geometry(points, axis_swap: bool = False) -> SpaceGeometry:
    """Centro, ángulo del eje, extensiones L_S y W_S y bbox de un polígono (se calcula una vez)."""
    poly = space_polygon(points)
    c = poly.centroid
    _, angle, short, long_ = min_area_rect_axis(points)
    breadth = short
    if axis_swap:
        angle, long_, breadth = (angle + 90.0) % 180.0, short, long_
    rad = math.radians(angle)
    dx, dy = math.cos(rad) * long_ / 2.0, math.sin(rad) * long_ / 2.0
    minx, miny, maxx, maxy = poly.bounds
    return SpaceGeometry((c.x, c.y), angle, maxx - minx, maxy - miny, (minx, miny, maxx, maxy),
                         ((c.x - dx, c.y - dy), (c.x + dx, c.y + dy)), long_, breadth)


@dataclass
class ParkingSpace:
    space_id: str
    points: list[tuple[float, float]]
    axis_swap: bool = False

    def polygon(self) -> Polygon:
        return space_polygon(self.points)

    def geometry(self) -> SpaceGeometry:
        return space_geometry(self.points, self.axis_swap)


@dataclass
class RuntimeSpace:
    """Puesto listo para el análisis (escalado a la resolución de procesamiento)."""
    space_id: str
    shape: Polygon                    # polígono Shapely
    polygon_img: np.ndarray           # (N, 2) int32 para dibujar
    geom: SpaceGeometry
    area: float
    aspect: float                     # ar_puesto = ancho / alto del bbox del puesto
    half_diag: float                  # ½ diagonal del bbox del puesto
    # Plano de medición: la imagen o, con "ground_quad", la vista cenital del piso (solo si es fiable: ver
    # ground_reliable). Si no, work_shape/work_geom caen a la imagen, igual que sin homografía.
    work_shape: Polygon | None = None
    work_geom: SpaceGeometry | None = None
    ground_reliable: bool = False

    def __post_init__(self):
        if self.work_shape is None:
            self.work_shape = self.shape
        if self.work_geom is None:
            self.work_geom = self.geom


class RuntimeLayout:
    def __init__(self, spaces: list[RuntimeSpace], size: tuple[int, int], homography: np.ndarray | None = None):
        self.spaces = spaces                                  # LISTA ordenada de puestos
        self.by_id = {s.space_id: s for s in spaces}          # DICT id -> puesto, O(1)
        self.size = size
        self.homography = homography                          # imagen -> vista cenital (None = medir en la imagen)


class ParkingLayout:
    def __init__(self, spaces: list[ParkingSpace] | None = None, reference_size: tuple[int, int] | None = None,
                 source: str = ""):
        self.spaces: list[ParkingSpace] = sorted(spaces or [], key=lambda s: natural_key(s.space_id))
        self.reference_size = reference_size
        self.source = source
        self.reference_thumbnail: str | None = None
        self.ground_quad: tuple[list[tuple[float, float]], tuple[float, float]] | None = None
        self.ground_angle: float = 90.0              # ángulo interior en la esquina 1 del piso (90 = rectángulo)
        self._cache: dict[tuple[int, int], RuntimeLayout] = {}

    # ---- edición ----------------------------------------------------------
    def get(self, space_id: str) -> ParkingSpace | None:
        return next((s for s in self.spaces if s.space_id == space_id), None)

    def next_id(self) -> str:
        used = {s.space_id for s in self.spaces}
        n = 1
        while f"P{n:02d}" in used:
            n += 1
        return f"P{n:02d}"

    def add(self, space: ParkingSpace) -> None:
        if self.get(space.space_id):
            raise ValueError(f"Ya existe un puesto {space.space_id}")
        self.spaces.append(space)
        self.spaces.sort(key=lambda s: natural_key(s.space_id))
        self.invalidate()

    def remove(self, space_id: str) -> None:
        self.spaces = [s for s in self.spaces if s.space_id != space_id]
        self.invalidate()

    def rename(self, old: str, new: str) -> None:
        new = new.strip().upper()
        if not new:
            raise ValueError("El nombre no puede estar vacío")
        if new != old and self.get(new):
            raise ValueError(f"Ya existe un puesto {new}")
        sp = self.get(old)
        if sp:
            sp.space_id = new
            self.spaces.sort(key=lambda s: natural_key(s.space_id))
        self.invalidate()

    def scale_points(self, new_size: tuple[int, int]) -> None:
        """Reescala los puntos si la imagen de calibración tiene otra resolución."""
        if self.reference_size and tuple(self.reference_size) != tuple(new_size):
            sx, sy = new_size[0] / self.reference_size[0], new_size[1] / self.reference_size[1]
            for sp in self.spaces:
                sp.points = [(x * sx, y * sy) for x, y in sp.points]
            if self.ground_quad:
                self.ground_quad = ([(x * sx, y * sy) for x, y in self.ground_quad[0]], self.ground_quad[1])
        self.reference_size = tuple(new_size)
        self.invalidate()

    def invalidate(self) -> None:
        self._cache.clear()

    def is_empty(self) -> bool:
        return not self.spaces

    # ---- persistencia -------------------------------------------------------
    def to_json(self) -> dict:
        spaces = {}
        for sp in self.spaces:
            g = sp.geometry()
            spaces[sp.space_id] = {
                "points": [[round(float(x), 1), round(float(y), 1)] for x, y in sp.points],
                "axis_swap": bool(sp.axis_swap),
                "center": [round(g.center[0], 1), round(g.center[1], 1)],
                "angle": round(g.angle, 2),
                "width": round(g.width, 1),
                "height": round(g.height, 1),
            }
        data = {"version": 4, "reference_resolution": list(self.reference_size) if self.reference_size else None,
                "units": "px", "source": self.source, "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reference_thumbnail": self.reference_thumbnail, "spaces": spaces}
        if self.ground_quad:
            pts, size = self.ground_quad
            data["ground_quad"] = {"image_points": [[round(float(x), 1), round(float(y), 1)] for x, y in pts],
                                   "size": [float(size[0]), float(size[1])],
                                   "angle": round(float(self.ground_angle), 2)}
        return data

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, indent=2, ensure_ascii=False)
        tmp.replace(path)
        log.info("Calibración guardada en %s (%d puestos)", path, len(self.spaces))

    @classmethod
    def load(cls, path: Path) -> "ParkingLayout":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        raw = data.get("spaces")
        if raw is None:                  # formato plano {"P01": {"points": [...]}}
            raw = {k: v for k, v in data.items() if isinstance(v, dict) and "points" in v}
        if data.get("ground_reference"):
            log.warning("%s tiene una 'referencia del piso' de una versión anterior: se ignora "
                        "(use 'ground_quad' para medir en vista cenital).", Path(path).name)
        spaces = []
        for sid, info in raw.items():
            pts = clean_polygon_points(info.get("points", []))
            if len(pts) < 3:
                log.warning("Puesto %s ignorado: menos de 3 puntos", sid)
                continue
            spaces.append(ParkingSpace(str(sid), pts, bool(info.get("axis_swap", False))))
        ref = data.get("reference_resolution")
        layout = cls(spaces, (int(ref[0]), int(ref[1])) if ref else None, data.get("source", ""))
        layout.reference_thumbnail = data.get("reference_thumbnail")
        layout.ground_quad = cls._parse_ground_quad(data.get("ground_quad"), Path(path).name)
        if layout.ground_quad:
            angle = cls._parse_ground_angle(data["ground_quad"], Path(path).name)
            if angle is None:
                layout.ground_quad = None
            else:
                layout.ground_angle = angle
        return layout

    @staticmethod
    def _parse_ground_angle(raw, name: str) -> float | None:
        """Ángulo del piso (90 si falta, como en los rectángulos); None si es inválido."""
        try:
            angle = float(raw.get("angle", 90.0))
            if not ANGLE_MIN <= angle <= ANGLE_MAX:
                raise ValueError(f"debe estar entre {ANGLE_MIN:g}° y {ANGLE_MAX:g}°")
            return angle
        except (TypeError, ValueError) as exc:
            log.warning("%s: 'ground_quad.angle' inválido (%s); se mide en la imagen (aproximado).", name, exc)
            return None

    @staticmethod
    def _parse_ground_quad(raw, name: str):
        """4 puntos del piso + tamaño real -> (puntos, (ancho, largo)); None si falta o es inválido."""
        if not raw:
            return None
        try:
            pts = [(float(p[0]), float(p[1])) for p in raw["image_points"]]
            size = (float(raw["size"][0]), float(raw["size"][1]))
            if len(pts) != 4 or size[0] <= 0 or size[1] <= 0:
                raise ValueError("se esperan 4 puntos y un tamaño positivo")
            problem = ground_quad_error(pts)
            if problem:
                raise ValueError(problem)
            return pts, size
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            log.warning("%s: 'ground_quad' inválido (%s); se mide en la imagen (aproximado).", name, exc)
            return None

    # ---- compatibilidad con una foto ------------------------------------------
    def check_compatibility(self, image_bgr: np.ndarray, min_similarity: float = 0.35) -> Compatibility:
        """¿Se pueden aplicar estos puestos a esta foto? (misma resolución/proporción y punto de vista)."""
        h, w = image_bgr.shape[:2]
        if self.is_empty():
            return Compatibility("none", "No hay puestos calibrados: solo detección y placa.")
        if not self.reference_size:
            return Compatibility("ok", "La calibración no guarda su resolución: se aplica tal cual.")
        rw, rh = self.reference_size
        ar_ref, ar_img = rw / float(rh), w / float(h)
        if abs(ar_img - ar_ref) / ar_ref > ASPECT_TOLERANCE:
            return Compatibility("mismatch", f"La foto ({w}×{h}) no tiene la misma proporción que la "
                                             f"calibración ({rw}×{rh}): los puestos no se pueden aplicar.")
        sim = viewpoint_similarity(self.reference_thumbnail, image_bgr) if self.reference_thumbnail else None
        if sim is not None and sim < min_similarity:
            return Compatibility("mismatch", f"La foto parece tomada desde otro PUNTO DE VISTA que la calibración "
                                             f"(similitud {sim:.2f} < {min_similarity:.2f}).", sim)
        note = "" if sim is not None else " (calibración sin imagen de referencia: no se pudo verificar el punto de vista)"
        if (w, h) == (rw, rh):
            return Compatibility("ok", f"Calibración aplicada ({w}×{h}){note}.", sim)
        return Compatibility("scaled", f"Puestos escalados de {rw}×{rh} a {w}×{h}{note}.", sim)

    # ---- versión para análisis ----------------------------------------------
    def runtime(self, width: int, height: int) -> RuntimeLayout:
        """Puestos escalados a (width, height). Se calcula una vez por resolución (caché)."""
        key = (int(width), int(height))
        if key in self._cache:
            return self._cache[key]
        ref_w, ref_h = self.reference_size or key
        sx, sy = width / float(ref_w), height / float(ref_h)
        homography = self._homography(sx, sy)
        ref_area = None
        if homography is not None and self.ground_quad:
            gw, gh = self.ground_quad[1]
            ref_area = gw * gh * math.sin(math.radians(self.ground_angle))
        out = []
        for sp in self.spaces:
            scaled = ParkingSpace(sp.space_id, [(x * sx, y * sy) for x, y in sp.points], sp.axis_swap)
            g = scaled.geometry()
            shape = scaled.polygon()
            minx, miny, maxx, maxy = g.bounds
            work_shape = work_geom = None
            reliable = False
            if homography is not None:
                ground_pts = transform_points(scaled.points, homography)
                cand_shape, cand_geom = space_polygon(ground_pts), space_geometry(ground_pts, sp.axis_swap)
                # La homografía se calibra con UN puesto; lejos de él, una lente con distorsión (fisheye) la
                # puede disparar (un puesto que mide ~2 m² sale midiendo miles). Si el área en el piso se aleja
                # demasiado del área real esperada, no se confía en el piso para ESTE puesto: se mide en la
                # imagen, como si no hubiera homografía (mismo aviso "ángulos aproximados").
                ratio = cand_shape.area / ref_area if ref_area and ref_area > 0 else None
                if ratio is not None and 1.0 / GROUND_AREA_TOLERANCE <= ratio <= GROUND_AREA_TOLERANCE:
                    work_shape, work_geom, reliable = cand_shape, cand_geom, True
            out.append(RuntimeSpace(sp.space_id, shape, np.round(np.asarray(scaled.points)).astype(np.int32), g,
                                    float(shape.area), aspect_ratio(minx, miny, maxx, maxy),
                                    0.5 * math.hypot(maxx - minx, maxy - miny), work_shape, work_geom, reliable))
        layout = RuntimeLayout(out, key, homography)
        self._cache[key] = layout
        return layout

    def _homography(self, sx: float, sy: float) -> np.ndarray | None:
        """Homografía imagen -> vista cenital a partir de los 4 puntos del piso (escalados a la resolución)."""
        if not self.ground_quad:
            return None
        pts, size = self.ground_quad
        src = np.array([(x * sx, y * sy) for x, y in pts], dtype=np.float32)
        dst = ground_destination(size, self.ground_angle, src).astype(np.float32)
        try:
            return cv2.getPerspectiveTransform(src, dst).astype(np.float64)
        except cv2.error:
            log.warning("No se pudo calcular la homografía del piso; se mide en la imagen (aproximado).")
            return None


def load_layout(path: Path) -> tuple[ParkingLayout | None, str]:
    """(layout o None, mensaje para el usuario). Nunca lanza excepciones."""
    if not path.exists():
        return None, "No hay puestos calibrados. Use 'Calibrar parqueadero'."
    try:
        layout = ParkingLayout.load(path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return None, f"El archivo de calibración está dañado ({exc}). Vuelva a calibrar."
    if layout.is_empty():
        return None, "La calibración no tiene puestos. Use 'Calibrar parqueadero'."
    return layout, f"Calibración cargada: {len(layout.spaces)} puestos ({path.name})."
