"""
core/pipeline.py — Orquestador del flujo VIMOP (no depende de la interfaz).

    FramePacket
      -> DETECCIÓN + TRACKING      YoloDetector.track (model.track, ByteTrack/BoT-SORT)
      -> ID                        IdentityManager  (MOTO #n, retención, re-identificación)
      -> IDENTIFICACIÓN            placa en IdentificationWorker (otro hilo)
      -> ANÁLISIS                  ParkingAnalyzer (línea de ejes R–F de WheelLocator, con Shapely)
      -> ESTADO                    MotionStateMachine (ENTRANDO/MOVIÉNDOSE/ESTACIONANDO/ESTACIONADA)
      -> PUNTUACIÓN/CLASIFICACIÓN  historial + histéresis (MotoRegistry)
      -> EVENTOS, ALERTAS, PERSISTENCIA (ResultsStore, asíncrono)
      -> FrameResult  (lo dibuja ui/renderer.py)

MODO FOTO (packet.static = True, fuente VideoSource("image", ...)):
    detección UNA vez (sin tracking) -> IDs secuenciales MOTO #1, #2... dentro de la foto
    -> sin máquina de estados: cada moto se evalúa directamente como ESTACIONADA
    -> placa sobre ese único frame, SIN votación de historial
    -> mismas fórmulas O, U, C, I, score y clasificación (sin histéresis)
    -> no se registran eventos de parqueo (no hay evolución temporal)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from analysis.parking_analyzer import INV_BIG, ParkingAnalysis, ParkingAnalyzer
from analysis.parking_spaces import ParkingLayout, RuntimeLayout
from analysis.wheels import WheelLocator
from analysis.state_machine import EVALUATED_STATES, ST_ENTERING, ST_MOVING, ST_PARKED, ST_PARKING
from core.camera import FramePacket
from core.detector import Detection, YoloDetector, select_device
from core.preprocess import prepare_frame
from core.registry import EV_ALERT, EV_GONE, MotoRegistry, ParkingEvent, score_row
from core.tracker import IdentityManager
from database import ResultsStore
from identification.worker import IdentificationWorker, IdentJob
from utils import CLASS_NAMES, classify_score, resize_max_width, safe_crop

log = logging.getLogger("vimop.pipeline")


@dataclass
class MotoView:
    info: dict
    detection: Detection | None
    analysis: ParkingAnalysis | None


@dataclass
class FrameResult:
    frame: np.ndarray
    frame_index: int
    timestamp: float
    wall_time: datetime
    persons: list[Detection] = field(default_factory=list)
    motos: list[MotoView] = field(default_factory=list)
    spaces: list[dict] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    alert_ids: set[int] = field(default_factory=set)
    new_alerts: list[str] = field(default_factory=list)
    events: list[ParkingEvent] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    inference_ms: float = 0.0
    process_ms: float = 0.0
    calibrated: bool = False
    static: bool = False                                  # True = MODO FOTO (análisis estático)
    origin: str = ""                                      # camara | video:<archivo> | foto:<archivo>
    notes: list[str] = field(default_factory=list)        # avisos (foto reducida, calibración escalada...)


class VimopPipeline:
    def __init__(self, cfg, layout: ParkingLayout | None, store: ResultsStore, status_cb=None):
        self.cfg = cfg
        self.store = store
        self.status = status_cb or (lambda msg: log.info(msg))
        self.layout = layout
        self.device, self.is_gpu, self.device_desc = select_device(cfg.DEVICE)
        self.status(f"Cargando YOLO en {self.device_desc}...")
        self.detector = YoloDetector(cfg, self.device, self.is_gpu)
        self.identities = IdentityManager(cfg)
        self.wheels = WheelLocator(cfg, self.device, self.is_gpu and bool(cfg.HALF_PRECISION))
        self.analyzer = ParkingAnalyzer(cfg, self.wheels)
        if not self.wheels.has_model and not self.detector.is_segmentation:
            log.warning("El detector '%s' no entrega máscaras y no hay %s: sin línea de ejes real, la orientación "
                        "se marcará no confiable. Use un modelo '-seg' o entrene wheels_pose.pt.",
                        self.detector.description, cfg.WHEEL_MODEL_PATH)
        self.registry = MotoRegistry(cfg)
        self.ident = IdentificationWorker(cfg, self.device, self.is_gpu)
        self.ident.start()
        self.pending_ident: set[int] = set()
        self.active_alerts: dict[tuple[int, str], str] = {}
        self.alert_last_seen: dict[tuple[int, str], float] = {}     # clave -> último instante activa
        self._prev_pos: dict[int, tuple[float, float]] = {}
        self._offset = np.zeros(2)
        self._last_log_t = -1e9
        self._last_autosave = time.monotonic()
        self.frame_count = 0
        self.origin = "camara"
        # Modo foto: una sola lectura por moto, sin votación (basta 1 lectura con confianza suficiente)
        self._static_cfg = cfg.copy()
        self._static_cfg.PLATE_MIN_VOTES = 1
        self._static_cfg.PLATE_MIN_CONSENSUS = 0.0

    def set_layout(self, layout: ParkingLayout | None) -> None:
        self.layout = layout

    def _runtime(self, w: int, h: int) -> RuntimeLayout | None:
        if self.layout is None or self.layout.is_empty():
            return None
        return self.layout.runtime(w, h)

    # ------------------------------------------------------------------
    def process(self, packet: FramePacket) -> FrameResult:
        if packet.static:
            return self._process_static(packet)
        t_start = time.perf_counter()
        cfg = self.cfg
        self.frame_count += 1
        fc = self.frame_count
        frame = prepare_frame(resize_max_width(packet.frame, int(cfg.PROCESS_MAX_WIDTH)), cfg)
        h, w = frame.shape[:2]
        t, now = packet.timestamp, packet.wall_time
        runtime = self._runtime(w, h)
        events: list[ParkingEvent] = []

        batch = self.detector.track(frame)                                 # DETECCIÓN + TRACKING
        trk = self.identities.update(batch.motorcycles, frame, t)          # ID
        offset = self._camera_offset(trk.tracked)

        self.registry.begin_frame()
        views: dict[int, MotoView] = {}
        for tm in trk.tracked:
            det = tm.detection
            rec = self.registry.observe(tm.moto_id, det.bbox, det.confidence, now)
            analysis = self.analyzer.analyze(det, runtime, frame) if runtime is not None else None   # ANÁLISIS

            # velocidad: punto medio M de la línea de ejes (estabilizado), en diagonales del bbox por segundo
            x1, y1, x2, y2 = det.bbox
            ref = analysis.center_img if analysis else ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            pos = tuple(np.asarray(ref, dtype=float) - offset)
            in_space = analysis is not None and analysis.space_id is not None
            events += self.registry.update_motion(rec, t, pos, in_space, max(det.diag, 1.0), now)   # ESTADO
            events += self.registry.apply_analysis(rec, analysis, t, now)                  # SCORE

            self._schedule_identification(rec, det, frame, fc)
            views[tm.moto_id] = MotoView({}, det, analysis)

        # conjuntos: motos activas que no se vieron en este frame
        self.registry.mark_hidden(self.registry.active_ids - self.registry.visible_ids)
        for moto_id in trk.finalized:
            rec = self.registry.finalize(moto_id)
            self.pending_ident.discard(moto_id)
            if rec is not None and rec.committed_class is not None:
                events.append(ParkingEvent(now, moto_id, EV_GONE, rec.committed_space, rec.smoothed_score(),
                                           rec.committed_class, rec.plate(cfg), "Salió del campo de visión"))

        for res in self.ident.drain():                                          # PLACA
            self.pending_ident.discard(res.moto_id)
            rec = self.registry.get(res.moto_id)
            if rec is None:
                continue
            if res.did_plate and res.plate is not None:
                events += self.registry.add_plate(rec, res.plate.text, res.plate.confidence, res.plate.method, now)

        motos_out = []
        for rec in self.registry.active():
            if not rec.visible and rec.state not in EVALUATED_STATES:
                continue                      # oculta y en movimiento: probablemente ya se fue
            view = views.get(rec.moto_id) or MotoView({}, None, rec.analysis)
            view.info = rec.to_dict(cfg)
            motos_out.append(view)

        alerts, alert_ids, new_alerts, alert_events = self._alerts(motos_out, now, t)
        events += alert_events
        for ev in events:
            self.store.log_event(ev)
        if t - self._last_log_t >= float(cfg.DETECTION_LOG_INTERVAL_SECONDS):
            self._last_log_t = t
            for v in motos_out:
                if v.info["estado_movimiento"] == ST_PARKED and v.info["puntaje"] is not None:
                    self.store.log_detection(v.info, now, packet.origin or self.origin)
        if time.monotonic() - self._last_autosave >= float(cfg.AUTOSAVE_SECONDS):
            self.save_all()

        return FrameResult(frame=frame, frame_index=packet.index, timestamp=t, wall_time=now,
                           persons=batch.persons, motos=motos_out, spaces=self._occupancy(runtime, motos_out),
                           alerts=alerts, alert_ids=alert_ids, new_alerts=new_alerts, events=events,
                           summary=self._summary(motos_out), inference_ms=batch.inference_ms,
                           process_ms=(time.perf_counter() - t_start) * 1000.0,
                           calibrated=runtime is not None, origin=packet.origin or self.origin)

    # ------------------------------------------------------------------
    def _process_static(self, packet: FramePacket) -> FrameResult:
        """MODO FOTO: análisis de una imagen fija (ver docstring del módulo)."""
        t_start = time.perf_counter()
        cfg, scfg = self.cfg, self._static_cfg
        frame = prepare_frame(packet.frame, cfg)
        h, w = frame.shape[:2]
        now = packet.wall_time
        runtime = self._runtime(w, h)
        batch = self.detector.detect(frame)                            # una sola detección, sin tracker
        if not self.ident.ready.is_set():
            self.status("Esperando a que cargue el OCR...")
            self.ident.ready.wait(timeout=180)
        registry = MotoRegistry(scfg)                                  # registro propio de esta foto
        views: list[MotoView] = []
        motos = sorted(batch.motorcycles, key=lambda d: (d.bbox[0], d.bbox[1]))   # orden de lectura estable
        for moto_id, det in enumerate(motos, start=1):                 # IDs secuenciales en la foto
            det.track_id = None
            rec = registry.observe(moto_id, det.bbox, det.confidence, now)
            rec.motion.state = ST_PARKED                               # sin movimiento que medir
            analysis = self.analyzer.analyze(det, runtime, frame) if runtime is not None else None
            rec.analysis = analysis
            if analysis is not None:
                rec.score_history.append(score_row(analysis))
                rec.space_history.append(analysis.space_id)
                rec.display_class = classify_score(analysis.score)     # sin histéresis
            crop, _ = safe_crop(frame, det.bbox, 0.05)
            if crop.size:
                if self.ident.plate_available and det.height >= int(cfg.PLATE_MIN_MOTO_HEIGHT_PX):
                    read = self.ident.plate_reader.read(crop)
                    if read is not None and read.confidence >= float(cfg.PLATE_MIN_CONFIDENCE):
                        registry.add_plate(rec, read.text, read.confidence, read.method, now)
            info = rec.to_dict(scfg)
            if analysis is None:
                info["estado_movimiento"] = "SIN PUESTOS"              # solo detección y placa
            views.append(MotoView(info, det, analysis))
        texts, ids, _ = self._alert_reasons(views)
        notes = [] if views else ["No se detectaron motocicletas."]
        return FrameResult(frame=frame, frame_index=0, timestamp=0.0, wall_time=now, persons=batch.persons,
                           motos=views, spaces=self._occupancy(runtime, views), alerts=texts, alert_ids=ids,
                           summary=self._summary(views), inference_ms=batch.inference_ms,
                           process_ms=(time.perf_counter() - t_start) * 1000.0, calibrated=runtime is not None,
                           static=True, origin=packet.origin, notes=notes)

    # ------------------------------------------------------------------
    def _camera_offset(self, tracked) -> np.ndarray:
        """
        Movimiento global acumulado de la imagen. Si la cámara vibra, todas las motos se
        desplazan igual: la mediana de sus desplazamientos es el movimiento de la cámara.
        Se aplica solo si el movimiento es coherente (≥ 60 % de las motos lo comparten).
        """
        cur = {tm.moto_id: tm.detection.bottom_center for tm in tracked}
        if bool(self.cfg.CAMERA_MOTION_COMPENSATION):
            deltas = np.array([np.subtract(p, self._prev_pos[i]) for i, p in cur.items() if i in self._prev_pos])
            if len(deltas) >= 3:
                med = np.median(deltas, axis=0)
                spread = np.linalg.norm(deltas - med, axis=1)
                if np.mean(spread <= max(3.0, 0.5 * float(np.linalg.norm(med)))) >= 0.6:
                    self._offset = self._offset + med
        self._prev_pos = cur
        return self._offset

    def _schedule_identification(self, rec, det: Detection, frame: np.ndarray, fc: int) -> None:
        cfg = self.cfg
        if not self.ident.ready.is_set() or rec.moto_id in self.pending_ident:
            return
        stable = rec.state in EVALUATED_STATES               # OCR solo con motos estables
        _, votes, _, _ = rec.plate_vote()
        want_plate = (self.ident.plate_available and stable and votes < int(cfg.PLATE_LOCK_VOTES)
                      and det.height >= int(cfg.PLATE_MIN_MOTO_HEIGHT_PX)
                      and fc - rec.last_ocr_frame >= int(cfg.OCR_INTERVAL))
        if not want_plate:
            return
        crop, _ = safe_crop(frame, det.bbox, 0.05)
        if crop.size == 0:
            return
        rec.last_ocr_frame = fc
        if self.ident.submit(IdentJob(rec.moto_id, crop.copy())):
            self.pending_ident.add(rec.moto_id)

    def _alerts(self, motos: list[MotoView], now: datetime, t: float):
        texts, ids, current = self._alert_reasons(motos)
        # alertas nuevas = actuales − anteriores (diferencia de conjuntos). Una alerta que
        # parpadea (moto justo en el límite) no se vuelve a registrar hasta ALERT_REARM_SECONDS.
        rearm = float(self.cfg.ALERT_REARM_SECONDS)
        new_keys = {k for k in set(current) - set(self.active_alerts)
                    if t - self.alert_last_seen.get(k, -1e9) > rearm}
        for k in current:
            self.alert_last_seen[k] = t
        self.active_alerts = current
        new_texts, events = [], []
        for moto_id, key in sorted(new_keys):
            rec, msg = self.registry.get(moto_id), current[(moto_id, key)]
            new_texts.append(f"⚠ MOTO #{moto_id:02d} — {msg}")
            if rec is not None:
                events.append(ParkingEvent(now, moto_id, EV_ALERT, rec.space(), rec.smoothed_score(),
                                           rec.display_class, rec.plate(self.cfg), msg))
        return texts, ids, new_texts, events

    def _alert_reasons(self, motos: list[MotoView]):
        """Motivos de alerta de cada moto ESTACIONADA: (textos, ids, {(id, motivo): mensaje})."""
        cfg = self.cfg
        current: dict[tuple[int, str], str] = {}
        texts, ids = [], set()
        for v in motos:
            info, a = v.info, v.analysis
            if info["estado_movimiento"] != ST_PARKED or info["puntaje"] is None:
                continue
            reasons = []
            if info["puesto"] == "FUERA" or (info["ubicacion"] or 0) < float(cfg.ALERT_MIN_LOCATION):
                reasons.append(("fuera", "FUERA DE PUESTO"))
            if a is not None and a.invasion_level == INV_BIG and a.invaded_spaces:
                reasons.append(("invasion", f"INVASIÓN {', '.join(a.invaded_spaces)}"))
            if info["puntaje"] < float(cfg.ALERT_MIN_SCORE):
                reasons.append(("score", f"SCORE BAJO {info['puntaje']:.1f}"))
            if reasons:
                ids.add(info["id"])
                texts.append(f"⚠ MOTO #{info['id']:02d} — PARQUEO INCORRECTO · " + " · ".join(r[1] for r in reasons))
                for key, msg in reasons:
                    current[(info["id"], key)] = msg
        return texts, ids, current

    @staticmethod
    def _occupancy(runtime: RuntimeLayout | None, motos: list[MotoView]) -> list[dict]:
        if runtime is None:
            return []
        occ: dict[str, list[int]] = {s.space_id: [] for s in runtime.spaces}
        for v in motos:
            if v.info["estado_movimiento"] in EVALUATED_STATES and v.info["puesto"] in occ:
                occ[v.info["puesto"]].append(v.info["id"])
        return [{"id": s.space_id, "polygon": s.polygon_img, "center": s.geom.center, "axis": s.geom.axis,
                 "occupants": occ[s.space_id]} for s in runtime.spaces]

    def _summary(self, motos: list[MotoView]) -> dict:
        per_class = {n: 0 for n in CLASS_NAMES}
        parked = parking = moving = correct = incorrect = 0
        for v in motos:
            info, st = v.info, v.info["estado_movimiento"]
            parked += st == ST_PARKED
            parking += st == ST_PARKING
            moving += st in (ST_MOVING, ST_ENTERING)
            if st == ST_PARKED and info["clasificacion"]:
                per_class[info["clasificacion"]] += 1
                if info["puntaje"] >= float(self.cfg.CORRECT_MIN_SCORE):
                    correct += 1
                else:
                    incorrect += 1
        return {"detectadas": len(motos), "estacionadas": parked, "estacionando": parking, "moviendose": moving,
                "correctas": correct, "incorrectas": incorrect, "por_clase": per_class}

    # ------------------------------------------------------------------
    def all_moto_dicts(self) -> list[dict]:
        return [rec.to_dict(self.cfg) for _, rec in sorted(self.registry.motos.items())]

    def save_all(self) -> None:
        self._last_autosave = time.monotonic()
        self.store.save_motos(self.all_moto_dicts())

    def shutdown(self) -> None:
        now = datetime.now()
        for moto_id in self.identities.finalize_all():
            rec = self.registry.finalize(moto_id)
            if rec is not None and rec.committed_class is not None:
                self.store.log_event(ParkingEvent(now, moto_id, EV_GONE, rec.committed_space, rec.smoothed_score(),
                                                  rec.committed_class, rec.plate(self.cfg), "Fin de la sesión"))
        self.save_all()
        self.ident.stop()
