"""
core/image_io.py — Lectura y escritura robusta de imágenes (modo FOTO).

- Rutas con tildes, ñ o espacios en Windows: se lee con np.fromfile + cv2.imdecode y se
  escribe con cv2.imencode + tofile (cv2.imread/imwrite fallan con algunas rutas Unicode).
- Canal alfa (PNG), escala de grises o 16 bits -> BGR de 3 canales y 8 bits.
- Orientación EXIF (fotos de celular): se lee la etiqueta 0x0112 y se rota la imagen.
- Fotos muy grandes: se reducen para que su lado mayor sea MAX_IMAGE_SIDE (y se informa).
- Tipo de archivo: por extensión y, si no es concluyente, por contenido.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("vimop.image_io")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# Filtros del selector nativo "EXAMINAR ARCHIVO" (ventana principal y menú inicial).
_VIDEO_GLOB = " ".join(f"*{e}" for e in VIDEO_EXTS)
_IMAGE_GLOB = " ".join(f"*{e}" for e in IMAGE_EXTS)
FILE_FILTERS = (f"Todos los soportados ({_VIDEO_GLOB} {_IMAGE_GLOB});;Videos ({_VIDEO_GLOB});;"
                f"Imágenes ({_IMAGE_GLOB})")


class ImageLoadError(RuntimeError):
    """El archivo no existe, está dañado o no es una imagen soportada (mensaje legible)."""


@dataclass
class LoadedImage:
    image: np.ndarray                     # BGR uint8 (H, W, 3)
    path: Path
    original_size: tuple[int, int]        # (ancho, alto) tras corregir EXIF, antes de reducir
    resized: bool = False
    exif_orientation: int = 1

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    def notes(self) -> list[str]:
        out = []
        if self.exif_orientation not in (0, 1):
            out.append(f"Orientación EXIF corregida (código {self.exif_orientation}).")
        if self.resized:
            w, h = self.original_size
            out.append(f"Foto grande ({w}×{h}) reducida a {self.size[0]}×{self.size[1]} píxeles.")
        return out


def exif_orientation(data: bytes) -> int:
    """Valor de la etiqueta EXIF Orientation (1–8). 1 si no hay EXIF o no se puede leer."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            value = int(im.getexif().get(0x0112, 1))
        return value if 1 <= value <= 8 else 1
    except Exception:          # sin Pillow, formato sin EXIF o EXIF dañado: no se rota
        return 1


def apply_exif_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
    """Transforma la imagen para que quede derecha según la etiqueta EXIF."""
    if orientation == 2:
        return cv2.flip(img, 1)
    if orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(img, 0)
    if orientation == 5:
        return cv2.transpose(img)
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.rotate(cv2.transpose(img), cv2.ROTATE_180)
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Cualquier imagen decodificada -> BGR uint8 de 3 canales."""
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img.astype(np.float64) * (255.0 if img.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        # la transparencia se compone sobre fondo blanco (no queda negra)
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        return (bgr * alpha + 255.0 * (1.0 - alpha)).round().astype(np.uint8)
    return img[:, :, :3].copy()


def decode_image(data: bytes) -> tuple[np.ndarray | None, int]:
    """Decodifica bytes de imagen. Devuelve (BGR o None, orientación EXIF aplicada)."""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)       # UNCHANGED: OpenCV no rota por EXIF
    if img is None:
        return None, 1
    orientation = exif_orientation(data)
    return apply_exif_orientation(to_bgr(img), orientation), orientation


def read_image(path: str | Path, max_side: int = 0) -> LoadedImage:
    """Lee una foto (ruta Unicode segura). Lanza ImageLoadError con un mensaje claro."""
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"No se encontró el archivo:\n{path}")
    try:
        data = np.fromfile(str(path), dtype=np.uint8).tobytes()
    except OSError as exc:
        raise ImageLoadError(f"No se pudo leer el archivo {path.name}: {exc}") from exc
    if not data:
        raise ImageLoadError(f"El archivo {path.name} está vacío.")
    img, orientation = decode_image(data)
    if img is None:
        raise ImageLoadError(f"El archivo {path.name} no es una imagen válida o está dañado.")
    h, w = img.shape[:2]
    loaded = LoadedImage(img, path, (w, h), False, orientation)
    if max_side and max(w, h) > int(max_side):
        s = int(max_side) / float(max(w, h))
        loaded.image = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        loaded.resized = True
    return loaded


def write_image(path: str | Path, img: np.ndarray, quality: int = 92) -> bool:
    """Guarda una imagen en una ruta Unicode. Devuelve False si falla (sin lanzar)."""
    path = Path(path)
    ext = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    try:
        ok, buf = cv2.imencode(ext, img, params)
        if not ok:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(path))
        return True
    except (cv2.error, OSError) as exc:
        log.error("No se pudo guardar %s: %s", path, exc)
        return False


def detect_file_kind(path: str | Path) -> str | None:
    """'image', 'video' o None (no soportado). Por extensión y, si no, por contenido."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if not path.is_file():
        return None
    try:                                             # contenido: ¿se decodifica como imagen?
        head = np.fromfile(str(path), dtype=np.uint8)
        if head.size and cv2.imdecode(head, cv2.IMREAD_UNCHANGED) is not None:
            return "image"
    except (OSError, cv2.error):
        return None
    cap = cv2.VideoCapture(str(path))                # ¿se abre como video?
    try:
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        return "video" if ok and frame is not None else None
    finally:
        cap.release()


def list_images_in_folder(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                  key=lambda p: p.name.lower())
