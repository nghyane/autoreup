#!/usr/bin/env python3
"""Voice audition — synthesize ONE sentence through several TTS setups so a human
can pick the voice/rate/pitch that sounds right. Evidence, not guesswork.

Writes data/auditions/*.m4a + an index.html with players, and tries to pull the
CapCut speaker catalog for future alternative voices.

Usage: .venv/bin/python scripts/audition_voice.py
"""

from __future__ import annotations

import html
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.core.config import Settings  # noqa: E402
from backend.app.pipeline.production import _build_capcut_client  # noqa: E402
from backend.app.services.audio import master_voice  # noqa: E402
from backend.app.services.tts import clean_tts_text  # noqa: E402
from backend.app.settings import SettingsStore, tts_binding  # noqa: E402

SAMPLE = (
    "Lão thần có nằm mơ cũng không ngờ, để trốn lệnh triệu hồi sang dị giới, "
    "nam chính lại đu bám lên tận trần lớp học. Vừa đặt chân tới nơi, cậu ngơ ngác chẳng hiểu mô tê gì."
)
OUT = ROOT / "data" / "auditions"


def fetch_catalog(settings: Settings, client) -> list[dict]:
    headers = dict(client._headers())  # noqa: SLF001 - audition tooling
    for path in ("/v2/speakers", "/v2/tts/speakers", "/v2/video/speakers"):
        try:
            request = urllib.request.Request(settings.capcut_edit_api + path, headers=headers)
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.load(response)
        except Exception as exc:  # noqa: BLE001
            print(f"catalog {path}: {exc}")
            continue
        speakers = (data.get("data") or {}).get("speakers") or data.get("speakers") or []
        if speakers:
            return [
                {
                    "speaker": s.get("speaker") or s.get("id"),
                    "name": s.get("name"),
                    "effect_id": s.get("effect_id"),
                    "resource_id": s.get("resource_id"),
                    "language": s.get("language") or s.get("lang"),
                }
                for s in speakers
            ]
    return []


def main() -> None:
    settings = Settings()
    app = SettingsStore(settings).load()
    task, conn = tts_binding(app)
    if conn is None or conn.type != "capcut":
        raise SystemExit("TTS connection is not CapCut (Cài đặt → Giọng đọc & ASR)")
    client = _build_capcut_client(settings, conn, task.voice)
    if client is None:
        raise SystemExit("no CapCut client (session/login failed)")
    OUT.mkdir(parents=True, exist_ok=True)

    variants = [
        ("TUNED: expressive text, rate 1.00, gentle master", dict(mode="expressive", rate=1.00, pitch=1.0, master=True)),
        ("expressive + rate 1.06 (post speed-up)", dict(mode="expressive", rate=1.06, pitch=1.0, master=True)),
        ("legacy natural text (all commas), rate 1.00", dict(mode="natural", rate=1.00, pitch=1.0, master=True)),
        ("expressive + RAW (no mastering)", dict(mode="expressive", rate=1.00, pitch=1.0, master=False)),
        ("expressive + pitch +3% (brighter)", dict(mode="expressive", rate=1.00, pitch=1.03, master=True)),
    ]

    entries: list[dict] = []
    for index, (label, options) in enumerate(variants):
        raw = OUT / f"{index:02d}.raw.m4a"
        try:
            text = clean_tts_text(SAMPLE, text_mode=options["mode"])
            raw.write_bytes(client.synthesize(text, rate=options["rate"], volume=10.0))
        except Exception as exc:  # noqa: BLE001
            print(f"skip [{label}]: {exc}")
            continue
        out = OUT / f"{index:02d}.m4a"
        if options["master"]:
            master_voice(raw, out, pitch=options["pitch"])
        else:
            out.write_bytes(raw.read_bytes())
        entries.append({"label": label, "file": out.name})
        print(f"ok [{label}] -> {out.name}")

    voices = fetch_catalog(settings, client)
    (OUT / "voices.json").write_text(json.dumps(voices, ensure_ascii=False, indent=2))
    print(f"catalog voices: {len(voices)}")
    vi_voices = [v for v in voices if (v.get("language") or "").lower().startswith("vi")]
    for voice in vi_voices[:10]:
        print("  vi voice:", voice)

    # Alternate speaker ids seen in the repo history — same effect/resource as BV421.
    # If CapCut accepts them, they widen the choice; a mismatch shows up as an error.
    original_speaker = client.config.speaker
    for index, (label, speaker) in enumerate(
        [("alt voice BV13x41117TL", "BV13x41117TL"), ("alt voice BV12N4y1M7rh", "BV12N4y1M7rh")],
        start=len(variants),
    ):
        client.config.speaker = speaker
        raw = OUT / f"{index:02d}.raw.m4a"
        try:
            raw.write_bytes(client.synthesize(text, rate=1.0, volume=10.0))
        except Exception as exc:  # noqa: BLE001
            print(f"skip [{label}]: {exc}")
            continue
        out = OUT / f"{index:02d}.m4a"
        master_voice(raw, out, pitch=1.0)
        entries.append({"label": label, "file": out.name})
        print(f"ok [{label}] -> {out.name}")
    client.config.speaker = original_speaker

    players = "\n".join(
        f'<li><p><b>{html.escape(entry["label"])}</b></p>'
        f'<audio controls preload="none" src="{html.escape(entry["file"])}"></audio></li>'
        for entry in entries
    )
    (OUT / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Voice audition</title>"
        "<body style='background:#101716;color:#e9f1ed;font:14px system-ui;padding:20px'>"
        f"<h2>Voice audition — {html.escape(settings.capcut_voice_name)} ({html.escape(settings.tts_voice)})</h2>"
        f"<ol style='line-height:2.4'>{players}</ol></body>",
        encoding="utf-8",
    )
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
