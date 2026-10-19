"""The user's own images (logo, watermark, sticker) stamped onto renders.

One shared library for every episode (``data/overlay_images``, served under /media):
a channel logo is uploaded once and placed per episode in the overlay editor. Every
upload is decoded and re-saved as an RGBA PNG, so only real images get in, transparency
survives, and the render never has to guess a format. An overlay region refers to an
image by its file NAME only — never a path."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

from backend.app.core.config import Settings

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_SIDE = 2048  # larger than any output frame: bigger only costs render time


class ImageUploadError(ValueError):
    pass


def images_dir(settings: Settings) -> Path:
    return settings.data_dir / "overlay_images"


def resolve_image(directory: Path | str, name: str) -> Path | None:
    """The library file called ``name``, or None (unknown, or not a bare file name)."""
    if not name or Path(name).name != name:
        return None
    path = Path(directory) / name
    return path if path.is_file() else None


def list_images(settings: Settings) -> list[dict]:
    """Newest first: what was just uploaded is what the editor wants next."""
    from PIL import Image

    directory = images_dir(settings)
    if not directory.is_dir():
        return []
    entries = []
    for path in sorted(directory.glob("*.png"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            with Image.open(path) as image:
                width, height = image.size
        except OSError:
            continue
        entries.append({"name": path.name, "width": width, "height": height})
    return entries


def save_image(settings: Settings, filename: str, data: bytes) -> dict:
    """Store an upload as ``<slug>-<hash>.png`` (the same image twice = one file)."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not data:
        raise ImageUploadError("file rỗng")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageUploadError(f"ảnh quá lớn (tối đa {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGBA")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ImageUploadError("không đọc được ảnh (dùng PNG, JPG hoặc WebP)") from error
    image.thumbnail((MAX_SIDE, MAX_SIDE))

    slug = re.sub(r"[^a-z0-9]+", "-", Path(filename or "").stem.lower()).strip("-")[:40] or "image"
    name = f"{slug}-{hashlib.sha1(data).hexdigest()[:8]}.png"
    directory = images_dir(settings)
    directory.mkdir(parents=True, exist_ok=True)
    image.save(directory / name)
    return {"name": name, "width": image.width, "height": image.height}


__all__ = ["ImageUploadError", "images_dir", "list_images", "resolve_image", "save_image"]
