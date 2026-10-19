
import backend.app.infrastructure.ffmpeg_encode as encode
from backend.app.core.config import Settings
from backend.app.pipeline.production import _hardsub_config
from backend.app.services.fonts import default_font, list_fonts, resolve_font
from backend.app.services.overlay import OverlayConfig



def _settings(tmp_path, **overrides) -> Settings:
    return Settings(root_dir=tmp_path, _env_file=None, **overrides)


def test_the_default_render_is_the_tier_that_looks_like_the_source(monkeypatch, tmp_path):
    # 2x a 441k AV1 source (881k) scored VMAF 79.7: a fixed measured tier, not a multiple
    monkeypatch.setattr(encode, "use_hw_encode", lambda settings: True)
    args = encode.video_encode_args(_settings(tmp_path))
    assert args[args.index("-b:v") + 1] == "2500000"
    assert next(p for p in encode.BITRATE_PRESETS if p["value"] == "2500k")["label"] == "Như gốc"

    monkeypatch.setattr(encode, "use_hw_encode", lambda settings: False)
    args = encode.video_encode_args(_settings(tmp_path), "4M")
    assert args[args.index("-maxrate") + 1] == "4000000" and "-crf" in args


def test_every_tier_is_a_rate_ffmpeg_accepts():
    for preset in encode.BITRATE_PRESETS:
        assert encode.valid_bitrate(preset["value"])
    assert encode.parse_bitrate("8M") == 8_000_000 and encode.parse_bitrate("900000") == 900_000


def test_the_overlay_font_is_burned_and_an_unknown_one_falls_back(tmp_path):
    settings = Settings(_env_file=None)  # the real repo: vendored fonts
    anton = next(font["path"] for font in list_fonts(settings) if font["name"] == "Anton")

    overlay = OverlayConfig.from_json(OverlayConfig(font=anton).to_json())
    assert _hardsub_config(settings, overlay).font_path == anton
    # never an arbitrary file from the request
    assert resolve_font(settings, "/etc/passwd") is None
    assert _hardsub_config(settings, OverlayConfig(font="/etc/passwd")).font_path == str(default_font(settings))
    assert _hardsub_config(settings, None).font_path == str(default_font(settings))


def test_only_fonts_with_vietnamese_glyphs_are_offered():
    names = {font["name"] for font in list_fonts(Settings(_env_file=None))}
    assert {"Anton", "Bangers"} <= names


def test_a_render_uses_the_bitrate_picked_in_the_editor(monkeypatch):
    monkeypatch.setattr(encode, "use_hw_encode", lambda settings: True)
    settings = Settings(_env_file=None)

    def rate(overlay):
        args = _hardsub_config(settings, overlay).video_codec_args
        return args[args.index("-b:v") + 1]

    assert rate(OverlayConfig.from_json({"bitrate": "4M"})) == "4000000"
    assert rate(OverlayConfig.from_json({"bitrate": ""})) == "2500000"  # the app setting
    # junk, out-of-range or the retired "source" mode fall back to the app setting
    for junk in ("8M; rm -rf /", "fast", "10k", "900M", "source"):
        assert OverlayConfig.from_json({"bitrate": junk}).bitrate == ""
    assert OverlayConfig.from_json(OverlayConfig(bitrate="1500k").to_json()).bitrate == "1500k"


def test_a_render_mixes_at_the_levels_picked_in_the_editor(tmp_path):
    from backend.app.services.audio import dub_mix_filter

    settings = _settings(tmp_path, dub_bed_volume=0.55)
    overlay = OverlayConfig.from_json(OverlayConfig(bed_volume=0.2, voice_volume=0.8).to_json())
    config = _hardsub_config(settings, overlay)
    assert (config.bed_volume, config.voice_volume) == (0.2, 0.8)
    graph = dub_mix_filter("[0:a:0]", "[2:a:0]", "[aout]", config.bed_volume, voice_volume=config.voice_volume)
    assert "volume=0.200" in graph and "volume=0.800" in graph

    # never touched in the editor: the app setting and a full-level voice
    untouched = _hardsub_config(settings, OverlayConfig.from_json({}))
    assert (untouched.bed_volume, untouched.voice_volume) == (0.55, 1.0)
    # a muted bed stays muted instead of falling back to the app setting
    assert _hardsub_config(settings, OverlayConfig.from_json({"bed_volume": 0})).bed_volume == 0.0
    # out of range clamps to 0-1, junk falls back
    clamped = OverlayConfig.from_json({"bed_volume": 3, "voice_volume": "loud"})
    assert (clamped.bed_volume, clamped.voice_volume) == (1.0, 1.0)
