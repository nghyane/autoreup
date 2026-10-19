"""Caption fonts the hardsub can burn: the vendored ones, ``data/fonts`` and the fonts
installed on this machine — only those that carry every Vietnamese glyph, since a
font missing "ự" burns a tofu box into every caption that uses it."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from backend.app.core.config import Settings

_VIETNAMESE = "ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹĂÂĐÊÔƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ"
_NOT_A_CHAR = "￿"  # never in a cmap: renders the font's .notdef glyph
_SUFFIXES = {".ttf", ".otf"}  # .ttc collections need a face index; not offered


@dataclass(frozen=True, slots=True)
class FontInfo:
    path: str
    name: str
    source: str  # app | user | system


def default_font(settings: Settings) -> Path:
    return Path(settings.hardsub_font) if settings.hardsub_font else settings.root_dir / "backend" / "app" / "assets" / "fonts" / "Bangers-Regular.ttf"


def font_dirs(settings: Settings) -> list[tuple[Path, str]]:
    home = Path.home()
    return [
        (settings.root_dir / "backend" / "app" / "assets" / "fonts", "app"),
        (settings.data_dir / "fonts", "app"),
        (home / "Library" / "Fonts", "user"),
        (Path("/Library/Fonts"), "system"),
        (Path("/System/Library/Fonts/Supplemental"), "system"),
        (home / ".local" / "share" / "fonts", "user"),
        (Path("/usr/share/fonts"), "system"),
    ]


def list_fonts(settings: Settings) -> list[dict]:
    """Every Vietnamese-capable font, app fonts first, then by name."""
    seen: set[str] = set()
    fonts: list[FontInfo] = []
    for directory, source in font_dirs(settings):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in _SUFFIXES:
                continue
            name = _vietnamese_font_name(str(path), path.stat().st_mtime)
            if name and name not in seen:
                seen.add(name)
                fonts.append(FontInfo(str(path), name, source))
    order = {"app": 0, "user": 1, "system": 2}
    fonts.sort(key=lambda font: (order[font.source], font.name.lower()))
    return [asdict(font) for font in fonts]


def resolve_font(settings: Settings, path: str) -> Path | None:
    """``path`` if it is one of the offered fonts (never an arbitrary file), else None."""
    if not path:
        return None
    return Path(path) if any(font["path"] == path for font in list_fonts(settings)) else None


@lru_cache(maxsize=2048)
def _vietnamese_font_name(path: str, _mtime: float) -> str | None:
    """"Family Style" when the font draws every Vietnamese letter, else None."""
    from PIL import ImageFont

    try:
        font = ImageFont.truetype(path, 24)
        notdef = _glyph(font, _NOT_A_CHAR)
        if any(_glyph(font, char) == notdef for char in _VIETNAMESE):
            return None
        family, style = font.getname()
    except (OSError, ValueError):
        return None
    family = (family or Path(path).stem).strip()
    style = (style or "").strip()
    return family if style.lower() in ("", "regular") else f"{family} {style}"


def _glyph(font, char: str) -> tuple:
    mask = font.getmask(char)
    return mask.size, bytes(mask)
