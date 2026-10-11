"""
core/photo.py — Modo FOTO sin interfaz: analizar una foto y exportar un lote.

La usan la interfaz (ui/photo_worker.py, en un hilo aparte) y la línea de comandos
(python main.py --headless --photos ...). Cada foto se abre con VideoSource("image", ruta),
la MISMA interfaz que las demás fuentes, y el pipeline la analiza en modo estático.

Política de calibración cuando la foto NO coincide con los puestos cargados:
    "ask"   -> no se analiza; se devuelve needs_decision=True para que el usuario elija
    "skip"  -> se analiza SIN puestos (solo detección y placa)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from analysis.parking_spaces import Compatibility, ParkingLayout
from core.camera import CameraError, VideoSource, origin_label
from core.pipeline import FrameResult, VimopPipeline
from database import batch_row, write_batch_csv

log = logging.getLogger("vimop.photo")


@dataclass
class PhotoOutcome:
    path: Path
    result: FrameResult | None = None
    frame: np.ndarray | None = None
    compat: Compatibility | None = None
    needs_decision: bool = False
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def calibration_label(self) -> str:
        if self.compat is None or self.compat.status == "none":
            return "sin calibración"
        return {"ok": "aplicada", "scaled": "escalada"}.get(self.compat.status, "no coincide: sin puestos")


def analyze_photo(pipeline: VimopPipeline, path: str | Path, layout: ParkingLayout | None,
                  policy: str = "ask") -> PhotoOutcome:
    """Analiza una foto. Nunca lanza excepciones: los problemas vuelven en outcome.error."""
    cfg = pipeline.cfg
    out = PhotoOutcome(Path(path))
    source = VideoSource("image", str(path), cfg)
    try:
        source.start()
        packet = source.read(timeout=1.0)
    except CameraError as exc:
        out.error = str(exc)
        return out
    finally:
        source.release()
    if packet is None:
        out.error = f"No se pudo leer la foto {out.path.name}."
        return out
    out.frame = packet.frame
    info = source.image_info
    out.notes = info.notes() if info is not None else []
    if layout is None or layout.is_empty():
        out.compat = Compatibility("none", "Sin puestos calibrados: solo detección y placa.")
    else:
        out.compat = layout.check_compatibility(packet.frame, float(cfg.VIEWPOINT_MIN_SIMILARITY))
    if out.compat.status == "mismatch" and policy == "ask":
        out.needs_decision = True
        return out
    pipeline.set_layout(layout if out.compat.usable else None)
    try:
        out.result = pipeline.process(packet)
    except Exception as exc:                  # una foto problemática no detiene la aplicación
        log.exception("Error analizando %s", out.path)
        out.error = f"Error analizando {out.path.name}: {exc}"
        return out
    out.notes = out.notes + [out.compat.message] + out.result.notes
    out.result.notes = out.notes
    return out


def export_batch(pipeline: VimopPipeline, paths: list[str | Path], layout: ParkingLayout | None,
                 csv_path: Path, progress: Callable[[int, int, str], None] | None = None,
                 cancelled: Callable[[], bool] | None = None,
                 on_outcome: Callable[[PhotoOutcome], None] | None = None) -> tuple[Path, int, int, list[str]]:
    """
    Procesa todas las fotos sin mostrarlas y escribe UN CSV resumen.
    Las fotos que no coinciden con la calibración se analizan sin puestos (se anota en el CSV).
    Devuelve (csv, fotos_procesadas, motos_totales, errores).
    """
    rows, errors, done, motos = [], [], 0, 0
    total = len(paths)
    for i, p in enumerate(paths, start=1):
        if cancelled is not None and cancelled():
            errors.append("Exportación cancelada por el usuario.")
            break
        if progress is not None:
            progress(i, total, Path(p).name)
        out = analyze_photo(pipeline, p, layout, policy="skip")
        if on_outcome is not None:
            on_outcome(out)
        when = datetime.now()
        if out.error:
            errors.append(out.error)
            rows.append({"Archivo": Path(p).name, "Origen": origin_label("image", p), "Calibracion": "",
                         "Estado": f"ERROR: {out.error.splitlines()[0]}", "Fecha": f"{when:%Y-%m-%d}",
                         "Hora": f"{when:%H:%M:%S}"})
            continue
        done += 1
        origin = out.result.origin or origin_label("image", p)
        if not out.result.motos:
            rows.append(batch_row(Path(p).name, origin, out.calibration_label, None, when))
        for view in out.result.motos:
            motos += 1
            rows.append(batch_row(Path(p).name, origin, out.calibration_label, view.info, when))
    write_batch_csv(csv_path, rows)
    return csv_path, done, motos, errors
