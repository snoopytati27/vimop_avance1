"""
core/camera.py — Captura de video con UNA sola clase: VideoSource.

    source = VideoSource("rtsp", "rtsp://...", cfg)     # o "webcam"/0, "ip"/URL, "video"/ruta, "image"/foto
    source.start()
    packet = source.read()        # FramePacket, o None (sin frame nuevo / video terminado)
    source.release()              # (alias de stop())

Un hilo lector toma los frames y los deja en una COLA (queue.Queue):
  - Fuente EN VIVO (webcam/ip/rtsp): cola de tamaño 1. Si llega un frame nuevo y el
    anterior no se ha procesado, el viejo se descarta -> siempre se procesa el
    frame MÁS RECIENTE y no se acumula latencia.
  - VIDEO: cola de tamaño 4 con put bloqueante -> no se descarta ningún frame
    (pruebas reproducibles), y el tiempo de cada frame sale del propio video.
  - IMAGE (foto): un único frame ESTÁTICO (packet.static = True). El pipeline lo analiza
    sin tracking ni máquina de estados. Se lee con core/image_io (rutas Unicode, alfa,
    escala de grises, EXIF, MAX_IMAGE_SIDE).
Si la cámara se cae, el hilo reintenta la conexión cada CAMERA_RECONNECT_SECONDS
(CAMERA_MAX_RETRIES = 0 -> para siempre) y publica el estado en `status`.
"""
from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.image_io import ImageLoadError, LoadedImage, read_image

import cv2
import numpy as np

log = logging.getLogger("vimop.camera")

SOURCE_TYPES = ("webcam", "ip", "rtsp", "video", "image")

ST_CONNECTING = "CONECTANDO"
ST_CONNECTED = "CONECTADA"
ST_PLAYING = "REPRODUCIENDO VIDEO"
ST_RECONNECTING = "RECONECTANDO"
ST_DISCONNECTED = "DESCONECTADA"
ST_ENDED = "VIDEO TERMINADO"
ST_FAILED = "SIN CONEXIÓN"
ST_PHOTO = "FOTO"


class CameraError(RuntimeError):
    """Error de apertura de la fuente con mensaje legible para el usuario."""


@dataclass
class FramePacket:
    frame: np.ndarray
    index: int              # número de frame
    timestamp: float        # segundos (tiempo del video o reloj monotónico)
    wall_time: datetime     # hora real de captura
    static: bool = False    # True = foto: análisis estático (sin tracking ni estados)
    origin: str = ""        # "camara", "video:<archivo>" o "foto:<archivo>"


def hide_password(url: str) -> str:
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        creds, host = rest.split("@", 1)
        return f"{scheme}://{creds.split(':', 1)[0]}:***@{host}"
    return url


def describe_source(kind: str, value: Any) -> str:
    if kind == "webcam":
        return f"Webcam {value}"
    if kind == "video":
        return f"Video: {Path(str(value)).name}"
    if kind == "image":
        return f"Foto: {Path(str(value)).name}"
    return f"{kind.upper()}: {hide_password(str(value))}"


def origin_label(kind: str, value: Any) -> str:
    """Valor de la columna Origen de detections.csv."""
    if kind == "video":
        return f"video:{Path(str(value)).name}"
    if kind == "image":
        return f"foto:{Path(str(value)).name}"
    return "camara"


def open_capture(kind: str, value: Any, cfg) -> cv2.VideoCapture:
    """Abre la fuente y comprueba que responda. Lanza CameraError con un mensaje claro."""
    kind = str(kind).lower()
    if kind not in SOURCE_TYPES:
        raise CameraError(f"SOURCE_TYPE desconocido: '{kind}'. Valores válidos: {', '.join(SOURCE_TYPES)}")
    if kind == "image":
        raise CameraError("Una foto no se abre como captura de video: use VideoSource('image', ruta, cfg).")

    if kind == "video":
        path = cfg.resolve_path(str(value))
        if not path.exists():
            raise CameraError(f"No se encontró el video:\n{path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise CameraError(f"OpenCV no pudo abrir el video (¿formato no soportado?):\n{path}")
        return cap

    if kind == "webcam":
        try:
            index = int(value)
        except (TypeError, ValueError):
            raise CameraError(f"Para webcam, CAMERA_SOURCE debe ser un número (0, 1, 2...). Recibido: {value!r}")
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap
            cap.release()
        raise CameraError(f"No se pudo abrir la webcam {index}.\n"
                          "Verifique que esté conectada y que ninguna otra aplicación la esté usando.")

    url = str(value).strip()
    if kind == "rtsp" and not url.lower().startswith("rtsp://"):
        raise CameraError("La URL RTSP debe comenzar por rtsp://")
    if kind == "ip" and not url.lower().startswith(("http://", "https://", "rtsp://")):
        raise CameraError("La URL de la cámara IP debe comenzar por http:// o https://")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{cfg.RTSP_TRANSPORT}"
    timeout = int(cfg.CAMERA_OPEN_TIMEOUT_MS)
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout])
    except (cv2.error, TypeError):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise CameraError(f"No se pudo conectar a la cámara {hide_password(url)}.\n"
                          "Revise IP, puerto, usuario/contraseña y que el PC esté en la misma red.")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except cv2.error:
        log.debug("El backend no permite ajustar el buffer de la cámara")
    return cap


def grab_single_frame(kind: str, value: Any, cfg, frame_index: int = 0) -> np.ndarray:
    """Un solo frame de la fuente (para calibrar)."""
    if kind == "image":
        try:
            return read_image(cfg.resolve_path(str(value)), int(cfg.MAX_IMAGE_SIDE)).image
        except ImageLoadError as exc:
            raise CameraError(str(exc)) from exc
    cap = open_capture(kind, value, cfg)
    try:
        if kind == "video" and frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        frame = None
        for _ in range(1 if kind == "video" else 5):     # los primeros frames en vivo suelen ser malos
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        if frame is None:
            raise CameraError("La fuente se abrió pero no entregó ningún frame.")
        return frame
    finally:
        cap.release()


class VideoSource:
    """Fuente de video única para webcam, cámara IP, RTSP y archivo."""

    def __init__(self, kind: str, value: Any, cfg):
        self.kind = str(kind).lower()
        self.value = value
        self.cfg = cfg
        self.is_live = self.kind not in ("video", "image")
        self.image_info: LoadedImage | None = None
        self.status = ST_CONNECTING
        self.message = ""
        self.fps = 30.0
        self.total_frames = 0
        # COLA productor (hilo lector) -> consumidor (inferencia)
        self._queue: queue.Queue[FramePacket] = queue.Queue(maxsize=1 if self.is_live else 4)
        self._stop = threading.Event()
        self._ended = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

    @classmethod
    def from_config(cls, cfg, video: bool = False) -> "VideoSource":
        if video:
            return cls("video", cfg.VIDEO_PATH, cfg)
        return cls(cfg.SOURCE_TYPE, cfg.CAMERA_SOURCE, cfg)

    def describe(self) -> str:
        return describe_source(self.kind, self.value)

    @property
    def finished(self) -> bool:
        """True cuando el video terminó (o la cámara agotó los reintentos) y la cola está vacía."""
        return self._ended.is_set() and self._queue.empty()

    @property
    def origin(self) -> str:
        return origin_label(self.kind, self.value)

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.kind == "image":
            self._start_image()
            return
        # Primer intento síncrono: un error de configuración se informa de inmediato.
        self._cap = open_capture(self.kind, self.value, self.cfg)
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1 <= fps <= 240 else (30.0 if not self.is_live else 25.0)
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if not self.is_live else 0
        self.status = ST_PLAYING if not self.is_live else ST_CONNECTED
        target = self._read_video if not self.is_live else self._read_live
        self._thread = threading.Thread(target=target, name="vimop-capture", daemon=True)
        self._thread.start()

    def _start_image(self) -> None:
        """Foto: se lee una vez y queda como único frame (sin hilo lector)."""
        try:
            self.image_info = read_image(self.cfg.resolve_path(str(self.value)), int(self.cfg.MAX_IMAGE_SIDE))
        except ImageLoadError as exc:
            raise CameraError(str(exc)) from exc
        self.fps, self.total_frames, self.status = 1.0, 1, ST_PHOTO
        self._queue.put(FramePacket(self.image_info.image, 0, 0.0, datetime.now(), static=True,
                                    origin=self.origin))
        self._ended.set()

    def read(self, timeout: float = 1.0) -> FramePacket | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def release(self) -> None:
        """Libera la fuente (mismo efecto que stop())."""
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        while not self._queue.empty():   # desbloquear un put() pendiente del hilo de video
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self.status not in (ST_ENDED, ST_FAILED, ST_PHOTO):
            self.status = ST_DISCONNECTED

    # ------------------------------------------------------------------
    def _put_latest(self, packet: FramePacket) -> None:
        """Deja solo el frame más reciente en la cola (fuentes en vivo)."""
        for _ in range(2):
            try:
                self._queue.put_nowait(packet)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()      # descartar el frame viejo y reintentar
                except queue.Empty:
                    continue

    def _read_live(self) -> None:
        index, t0, failures, retries = -1, time.monotonic(), 0, 0
        max_retries = int(self.cfg.CAMERA_MAX_RETRIES)
        while not self._stop.is_set():
            if self._cap is None:
                self.status = ST_RECONNECTING
                try:
                    self._cap = open_capture(self.kind, self.value, self.cfg)
                    self.status, self.message, retries = ST_CONNECTED, "", 0
                    log.info("Cámara reconectada: %s", self.describe())
                except CameraError as exc:
                    retries += 1
                    self.message = str(exc)
                    if max_retries and retries >= max_retries:
                        self.status = ST_FAILED
                        self.message = f"Cámara no disponible tras {retries} intentos. {exc}"
                        self._ended.set()
                        return
                    self._stop.wait(float(self.cfg.CAMERA_RECONNECT_SECONDS))
                    continue
            ok, frame = self._cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= 25:
                    log.warning("La cámara dejó de responder; reconectando...")
                    self.message = "La cámara dejó de enviar video (¿desconectada o RTSP caído?). Reconectando..."
                    self._cap.release()
                    self._cap = None
                    failures = 0
                else:
                    time.sleep(0.02)
                continue
            failures = 0
            index += 1
            self._put_latest(FramePacket(frame, index, time.monotonic() - t0, datetime.now(), origin="camara"))

    def _read_video(self) -> None:
        cfg = self.cfg
        index, loop_offset, t0 = -1, 0.0, time.monotonic()
        step = max(1, int(cfg.VIDEO_FRAME_STEP))
        while not self._stop.is_set():
            frame = None
            for _ in range(step):
                ok, f = self._cap.read()
                if (not ok or f is None) and cfg.VIDEO_LOOP and index > 0:
                    loop_offset += (index + 1) / self.fps
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    index = -1
                    ok, f = self._cap.read()
                if not ok or f is None:
                    break
                index += 1
                frame = f
            if frame is None:
                self.status = ST_ENDED
                self.message = "El video terminó."
                self._ended.set()
                return
            ts = loop_offset + index / self.fps
            if cfg.VIDEO_REALTIME:
                delay = (t0 + ts) - time.monotonic()
                if delay > 0:
                    time.sleep(min(delay, 0.5))
                elif delay < -1.0:          # el procesamiento va más lento que el video
                    t0 = time.monotonic() - ts
            packet = FramePacket(frame, index, ts, datetime.now(), origin=self.origin)
            while not self._stop.is_set():  # put bloqueante (sin perder frames) pero interrumpible
                try:
                    self._queue.put(packet, timeout=0.2)
                    break
                except queue.Full:
                    continue
