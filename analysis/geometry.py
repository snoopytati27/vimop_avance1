"""
analysis/geometry.py — Geometría pura (sin estado).

MODELO: cada moto es un SEGMENTO R–F entre el punto de contacto de la rueda trasera y el de la
delantera con el piso (línea de ejes). Los puestos son polígonos calibrados. Las medidas se hacen
en la imagen (píxeles) o, si la calibración trae 4 puntos del piso, en la vista cenital.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid


def space_polygon(points) -> Polygon:
    """Polígono de un puesto; si el usuario cruzó los vértices, se repara con make_valid."""
    poly = Polygon([(float(x), float(y)) for x, y in points])
    if not poly.is_valid:
        fixed = make_valid(poly)
        polys = [g for g in getattr(fixed, "geoms", [fixed]) if isinstance(g, Polygon)]
        poly = max(polys, key=lambda g: g.area) if polys else poly.buffer(0)
    return poly


def aspect_ratio(minx: float, miny: float, maxx: float, maxy: float) -> float:
    """ancho / alto de un rectángulo alineado a los ejes."""
    return max(maxx - minx, 1e-9) / max(maxy - miny, 1e-9)


def axis_angle_deg(vx: float, vy: float) -> float:
    """Ángulo de un EJE sin sentido, en [0, 180)."""
    return math.degrees(math.atan2(vy, vx)) % 180.0


def angular_error_deg(a: float, b: float) -> float:
    """d = |a − b| mod 180 ; err = min(d, 180 − d) ∈ [0, 90]  (0° ≡ 180°)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def principal_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """PCA 2D: (centro, eje_mayor_unitario, varianza_mayor, elongación = sqrt(λ1/λ2))."""
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T) if len(pts) > 2 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    lam1, lam2 = float(max(evals[1], 1e-12)), float(max(evals[0], 1e-12))
    return center, evecs[:, 1], lam1, math.sqrt(lam1 / lam2)


def min_area_rect_axis(points) -> tuple[np.ndarray, float, float, float]:
    """cv2.minAreaRect: (centro, ángulo del lado largo, lado corto, lado largo)."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    b = cv2.boxPoints(cv2.minAreaRect(pts))
    e1, e2 = b[1] - b[0], b[2] - b[1]
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    edge = e1 if l1 >= l2 else e2
    return b.mean(axis=0).astype(np.float64), axis_angle_deg(float(edge[0]), float(edge[1])), min(l1, l2), max(l1, l2)


def polygon_centroid(points) -> tuple[float, float]:
    c = space_polygon(points).centroid
    return float(c.x), float(c.y)


def wheel_axis_from_mask(mask: np.ndarray, origin: tuple[float, float], bottom_band: float = 0.25,
                         inset: float = 0.12) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Respaldo sin entrenar para ubicar las ruedas con la máscara de segmentación (bool, recortada
    al bbox cuya esquina superior izquierda está en `origin`).

    1. Toma la franja inferior de la máscara (`bottom_band` de su altura).
    2. Eje principal de esos puntos con PCA.
    3. Extremos del eje, metidos `inset` de la longitud hacia adentro: el extremo de la silueta es
       el borde de la rueda y no su punto de contacto con el piso.
    Devuelve (R, F) ordenados de izquierda a derecha en la imagen, o None si la máscara no alcanza.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) < 8:
        return None
    top, bottom = float(ys.min()), float(ys.max())
    band = ys >= bottom - max(1.0, bottom_band * (bottom - top + 1.0))
    if int(band.sum()) < 8:
        return None
    pts = np.stack([xs[band] + origin[0], ys[band] + origin[1]], axis=1).astype(np.float64)
    center, v, _, _ = principal_axis(pts)
    if v[0] < 0 or (abs(v[0]) < 1e-9 and v[1] < 0):      # sentido estable: de izquierda a derecha
        v = -v
    t = (pts - center) @ v
    t_min, t_max = float(np.percentile(t, 2)), float(np.percentile(t, 98))     # robusto a píxeles sueltos
    length = t_max - t_min
    if length < 2.0:
        return None
    t_min, t_max = t_min + inset * length, t_max - inset * length
    return center + t_min * v, center + t_max * v


def bbox_axis(bbox, bottom_band: float = 0.25, inset: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """
    Último recurso, sin máscara ni modelo de ruedas: segmento a lo largo del lado MAYOR del bbox.
    Un bbox solo da 0° o 90°, así que quien lo use debe marcar la orientación como no confiable.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = x2 - x1, y2 - y1
    if w >= h:
        y = y2 - 0.5 * bottom_band * h
        return np.array([x1 + inset * w, y]), np.array([x2 - inset * w, y])
    x = (x1 + x2) / 2.0
    return np.array([x, y1 + inset * h]), np.array([x, y2 - inset * h])


def segment_axis_angle_deg(r, f) -> float:
    """Ángulo del eje R→F en [0, 180)."""
    return axis_angle_deg(float(f[0] - r[0]), float(f[1] - r[1]))


def transform_points(points, homography: np.ndarray | None) -> np.ndarray:
    """Aplica una homografía 3x3 a puntos (N, 2); sin homografía devuelve los puntos tal cual."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if homography is None:
        return pts
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.asarray(homography, dtype=np.float64)).reshape(-1, 2)
