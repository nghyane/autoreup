import io
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import backend.app.api.v1.endpoints.settings as settings_endpoint
from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.pipeline.production import _hardsub_config
from backend.app.services.audio import ffmpeg_available
from backend.app.services.hardsub import _cover_regions, build_hardsub, render_overlay_frame
from backend.app.services.overlay import CoverRegion, OverlayConfig
from backend.app.services.overlay_images import ImageUploadError, images_dir, list_images, resolve_image, save_image

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")


FONT = Path(__file__).resolve().parents[1] / "backend" / "app" / "assets" / "fonts" / "Bangers-Regular.ttf"


def _settings(tmp_path) -> Settings:
    settings = Settings(root_dir=tmp_path, _env_file=None, video_width=320, video_height=240, hardsub_font=str(FONT))
    settings.ensure_directories()
    return settings


def _png(size=(40, 20), color=(255, 0, 0, 255), fmt="PNG") -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGBA", size, color)
    (image.convert("RGB") if fmt == "JPEG" else image).save(buffer, fmt)
    return buffer.getvalue()


def test_an_upload_is_stored_as_a_png_that_keeps_its_transparency(tmp_path):
    settings = _settings(tmp_path)
    saved = save_image(settings, "Kênh Của Tôi.png", _png(color=(255, 0, 0, 128)))
    assert saved["name"].startswith("k-nh-c-a-t-i-") and saved["name"].endswith(".png")
    with Image.open(images_dir(settings) / saved["name"]) as image:
        assert image.mode == "RGBA" and image.getpixel((0, 0))[3] == 128
    # the same picture twice is one file; a JPEG becomes a PNG too
    assert save_image(settings, "Kênh Của Tôi.png", _png(color=(255, 0, 0, 128)))["name"] == saved["name"]
    assert save_image(settings, "photo.jpg", _png(fmt="JPEG"))["name"].endswith(".png")
    assert [image["name"] for image in list_images(settings)][1] == saved["name"]  # newest first


def test_only_real_images_get_into_the_library(tmp_path):
    settings = _settings(tmp_path)
    for junk in (b"", b"not an image", b"<svg></svg>"):
        with pytest.raises(ImageUploadError):
            save_image(settings, "x.png", junk)
    huge = save_image(settings, "big.png", _png(size=(5000, 1000)))
    assert (huge["width"], huge["height"]) == (2048, 410)


def test_a_region_names_an_image_never_a_path(tmp_path):
    settings = _settings(tmp_path)
    name = save_image(settings, "logo.png", _png())["name"]
    assert resolve_image(images_dir(settings), name) is not None
    for bad in ("../library.db", "/etc/passwd", "sub/logo.png", "", "missing.png"):
        assert resolve_image(images_dir(settings), bad) is None
    stored = OverlayConfig.from_json({"cover": [{"x": 0, "y": 0, "w": 0.2, "h": 0.2, "mode": "image", "image": "../../" + name}]})
    assert stored.to_json()["cover"][0]["image"] == name
    # a cover box never carries an image name
    cover = OverlayConfig.from_json({"cover": [{"x": 0, "y": 0, "w": 0.2, "h": 0.2, "mode": "blur", "image": name}]})
    assert cover.to_json()["cover"][0]["image"] == ""


def test_the_editor_uploads_and_lists_images(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(settings_endpoint, "get_settings", lambda: settings)
    client = TestClient(create_app())
    response = client.post("/api/v1/settings/images", files={"file": ("logo.png", _png(), "image/png")})
    assert response.status_code == 200, response.text
    assert response.json()["width"] == 40
    listing = client.get("/api/v1/settings/images").json()
    assert listing["base"] == "/media/overlay_images" and listing["images"][0]["name"] == response.json()["name"]
    bad = client.post("/api/v1/settings/images", files={"file": ("x.png", b"nope", "image/png")})
    assert bad.status_code == 400 and "ảnh" in bad.json()["detail"]


def _image_overlay(settings, strength=1.0, image=None) -> OverlayConfig:
    name = image or save_image(settings, "logo.png", _png(size=(40, 40)))["name"]
    # a 40x40 red square fitted into a 0.5 x 0.5 box at the top-left of a 320x240 frame:
    # 120x120, centred horizontally in the 160px-wide box -> x 20..140, y 0..120
    return OverlayConfig(caption_mode="manual", caption_y=0.9, cover=[
        CoverRegion(x=0.0, y=0.0, w=0.5, h=0.5, mode="image", strength=strength, label="image", image=name),
    ])


def _navy_video(path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1.5", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def _pixel(path, xy, t=None) -> tuple[int, int, int]:
    if t is not None:
        frame = path.with_suffix(".png")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1", str(frame)], check=True)
        path = frame
    with Image.open(path) as image:
        return image.convert("RGB").getpixel(xy)


def test_image_regions_are_not_covers(tmp_path):
    config = _hardsub_config(_settings(tmp_path), _image_overlay(_settings(tmp_path)))
    assert _cover_regions(None, config, 320, 240) == []


@needs_ffmpeg
def test_the_preview_frame_shows_the_image_inside_its_box(tmp_path):
    settings = _settings(tmp_path)
    video, frame = tmp_path / "v.mp4", tmp_path / "frame.png"
    _navy_video(video)
    render_overlay_frame(video, frame, 0.5, _hardsub_config(settings, _image_overlay(settings)))
    red, green, blue = _pixel(frame, (80, 60))
    assert red > 200 and green < 60 and blue < 60
    assert max(_pixel(frame, (10, 60))) < 40  # the box's empty side: the picture shows through
    assert max(_pixel(frame, (250, 200))) < 40

    render_overlay_frame(video, frame, 0.5, _hardsub_config(settings, _image_overlay(settings, strength=0.5)))
    assert 90 < _pixel(frame, (80, 60))[0] < 170  # half opacity over black


@needs_ffmpeg
def test_the_render_stamps_the_image_for_the_whole_video(tmp_path):
    settings = _settings(tmp_path)
    video, out = tmp_path / "v.mp4", tmp_path / "out.mp4"
    _navy_video(video)
    cues = [{"start": 0.2, "end": 0.6, "text": "xin chào"}]
    build_hardsub(video, cues, None, out, tmp_path / "work", _hardsub_config(settings, _image_overlay(settings)))
    for t in (0.1, 0.4, 1.2):  # before, during and after the caption
        assert _pixel(out, (80, 60), t=t)[0] > 200

    # deleted from the library: the render goes on without it
    missing = _hardsub_config(settings, _image_overlay(settings, image="gone.png"))
    build_hardsub(video, [], None, out, tmp_path / "work2", missing)
    assert max(_pixel(out, (80, 60), t=0.5)) < 40
