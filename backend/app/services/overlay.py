"""Overlay config — what the viewer should (not) see, editable per job.

The package render is a deterministic function of (video, cues, voice, overlay). This
module owns the LAST input: where the Vietnamese caption sits and which regions of the
picture get covered (the original subtitle band, a channel logo, a watermark).

Coordinates are FRACTIONS of the frame (0..1), never pixels: a config must survive any
source resolution — the same lesson as the output-space caption fix (a 4K source must
not produce different-looking output than a 1080p one).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

COVER_MODES = ("blur", "darken", "solid", "image")


@dataclass(slots=True)
class CoverRegion:
    """One rectangle to hide (fractions of width/height), plus how to hide it — or,
    with mode "image", where to stamp one of the user's images (services.overlay_images)."""

    x: float
    y: float
    w: float
    h: float
    mode: str = "blur"        # blur | darken | solid | image
    strength: float = 1.0     # blur: radius scale; darken/solid/image: 0..1 opacity
    label: str = ""           # "band" | "logo" | free text
    image: str = ""           # mode "image": the library file name (fitted inside the box)

    def normalized(self) -> "CoverRegion":
        x = min(max(float(self.x), 0.0), 1.0)
        y = min(max(float(self.y), 0.0), 1.0)
        w = min(max(float(self.w), 0.01), 1.0 - x)
        h = min(max(float(self.h), 0.01), 1.0 - y)
        mode = self.mode if self.mode in COVER_MODES else "blur"
        image = Path(str(self.image or "")).name if mode == "image" else ""
        return CoverRegion(round(x, 4), round(y, 4), round(w, 4), round(h, 4), mode,
                           round(min(max(float(self.strength), 0.05), 1.0), 3), self.label, image)


@dataclass(slots=True)
class OverlayConfig:
    # band  = the caption follows the region labelled "band" (the original subtitle),
    # manual = the caption sits at caption_y (fraction from the top)
    caption_mode: str = "band"
    caption_y: float | None = None
    caption_scale: float = 1.0
    cover: list[CoverRegion] = field(default_factory=list)
    font: str = ""  # caption font file (one of services.fonts.list_fonts); "" = the app default
    bitrate: str = ""  # video bitrate for this render ("2500k", "4M"); "" = the app setting
    bed_volume: float | None = None  # original audio under the dub (0-1); None = the app setting
    voice_volume: float = 1.0  # the dub itself (0-1)
    gap_volume: float | None = None  # original audio where nobody speaks (0-1); None = the app setting

    def band_region(self) -> CoverRegion | None:
        for region in self.cover:
            if region.label == "band":
                return region
        return None

    def caption_cy_fraction(self) -> float | None:
        """Vertical centre of the caption as a fraction, or None when it must be derived."""
        if self.caption_mode == "manual" and self.caption_y is not None:
            return min(max(float(self.caption_y), 0.0), 1.0)
        band = self.band_region()
        if band is not None:
            return min(max(band.y + band.h / 2, 0.0), 1.0)
        return None

    def to_json(self) -> dict:
        return {
            "caption_mode": self.caption_mode if self.caption_mode in ("band", "manual") else "band",
            "caption_y": None if self.caption_y is None else round(float(self.caption_y), 4),
            "caption_scale": round(min(max(float(self.caption_scale), 0.5), 2.0), 3),
            "cover": [asdict(region.normalized()) for region in self.cover],
            "font": str(self.font or ""),
            "bitrate": _bitrate(self.bitrate),
            "bed_volume": _volume(self.bed_volume),
            "voice_volume": _volume(self.voice_volume, 1.0),
            "gap_volume": _volume(self.gap_volume),
        }

    @classmethod
    def from_json(cls, data: dict | None) -> "OverlayConfig":
        data = data or {}
        regions = [CoverRegion(**{k: v for k, v in item.items() if k in CoverRegion.__slots__}) for item in data.get("cover") or []]
        return cls(
            caption_mode=str(data.get("caption_mode") or "band"),
            caption_y=(None if data.get("caption_y") is None else float(data["caption_y"])),
            caption_scale=float(data.get("caption_scale") or 1.0),
            cover=regions,
            font=str(data.get("font") or ""),
            bitrate=_bitrate(data.get("bitrate")),
            bed_volume=_volume(data.get("bed_volume")),
            voice_volume=_volume(data.get("voice_volume"), 1.0),
            gap_volume=_volume(data.get("gap_volume")),
        )


def _bitrate(value) -> str:
    """Keep only a rate ffmpeg can take; anything else means "use the app setting"."""
    from backend.app.infrastructure.ffmpeg_encode import valid_bitrate

    text = str(value or "").strip()
    return text if text and valid_bitrate(text) else ""


def _volume(value, default: float | None = None) -> float | None:
    """A gain in 0-1 (rounded); anything unreadable falls back to ``default``."""
    try:
        return round(min(max(float(value), 0.0), 1.0), 3)
    except (TypeError, ValueError):
        return default


def overlay_path(job_dir: Path, job_id: str) -> Path:
    return Path(job_dir) / "overlay.json"


def load_overlay(job_dir: Path, job_id: str) -> OverlayConfig | None:
    path = overlay_path(job_dir, job_id)
    if not path.exists():
        return None
    try:
        return OverlayConfig.from_json(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def save_overlay(job_dir: Path, job_id: str, config: OverlayConfig) -> OverlayConfig:
    path = overlay_path(job_dir, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = OverlayConfig.from_json(config.to_json())
    path.write_text(json.dumps(normalized.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def default_overlay(video: Path) -> OverlayConfig:
    """A fresh job starts from what the detector sees: the original subtitle band.

    The band becomes an editable region (label "band") so the UI shows exactly what
    will be covered, and the caption anchors to that region's centre."""
    from backend.app.services.hardsub import HardsubConfig, detect_sub_band

    config = HardsubConfig()
    band = detect_sub_band(Path(video), config)
    import cv2

    cap = cv2.VideoCapture(str(video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap.release()
    if band is None:
        return OverlayConfig(caption_mode="manual", caption_y=round(1 - config.margin_v / max(1, height), 4), cover=[])
    band_y, band_h, _w, _h = band
    return OverlayConfig(
        caption_mode="band",
        cover=[CoverRegion(x=0.0, y=round(band_y / height, 4), w=1.0, h=round(band_h / height, 4),
                           mode="blur", strength=1.0, label="band")],
    )


__all__ = ["COVER_MODES", "CoverRegion", "OverlayConfig", "default_overlay", "load_overlay", "overlay_path", "save_overlay"]
