"""JSON settings store at ``.autoreup-private/settings.json`` (never served).

No file -> the settings are SEEDED from the env (``AUTOREUP_*``), so an install that
never opened the settings UI behaves exactly as before. The file is read on every
``load()`` (no cache): a change applies to the next job without a restart, while a
running job keeps the secret-free snapshot taken when it was created.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from backend.app.core.config import Settings

from pydantic import ValidationError

from .schema import SECRET_FIELDS, SNAPSHOT_PRIVATE_FIELDS, AppSettings, AsrTask, ChatTask, Connection, OcrTask, Tasks, TtsTask

MASK = "••••"
_LOCK = threading.Lock()


class SettingsFileError(ValueError):
    """settings.json exists but cannot be read; the message never echoes file content."""


def mask_secret(value: str) -> str:
    if not value:
        return ""
    return MASK + value[-4:] if len(value) > 8 else MASK


def _is_placeholder(value: str) -> bool:
    return not value or value.startswith(MASK)


def seed_from_env(env: Settings) -> AppSettings:
    """The settings an env-only install implicitly had."""
    provider = (env.agent_provider or "agy").strip().lower()
    if provider in {"command", "cli", "acp"}:
        agent_type = "command"
    elif provider in {"claude", "claude_code"}:
        agent_type = "claude_code"
    elif provider == "codex":
        agent_type = "codex"
    else:
        agent_type = "agy"
    agent = Connection(
        id="agent",
        label="Agent CLI",
        type=agent_type,
        # AGENT_BIN defaults to "agy": never hand that binary to another CLI
        bin="" if agent_type != "agy" and env.agent_bin == "agy" else env.agent_bin,
        home=env.agent_home or "",
        command=env.agent_command,
    )
    if env.tts_native:
        tts_conn = Connection(
            id="capcut", label="CapCut", type="capcut", base_url=env.tts_server_url or "",
            email=env.capcut_email, password=env.capcut_password,
        )
    else:
        tts_conn = Connection(id="tts-server", label="Máy chủ TTS", type="http_tts", base_url=env.tts_server_url or "")

    def chat(model: str) -> ChatTask:
        return ChatTask(connection=agent.id, model=model, effort=env.agent_effort)

    return AppSettings(
        connections=[agent, tts_conn],
        tasks=Tasks(
            translate=chat(env.translate_model),
            glossary=chat(env.translate_model),
            normalize=chat(env.translate_model),
            repair=chat(env.polish_model),
            naming=chat(env.translate_model),
            ocr=OcrTask(recognizer="vision"),
            tts=TtsTask(enabled=env.tts_enabled, connection=tts_conn.id, voice=env.tts_voice, synth_rate=env.tts_synth_rate),
            asr=AsrTask(
                backend=env.asr_backend if env.asr_backend in {"auto", "qwen", "local", "worker"} else "auto",
                worker_url=env.asr_worker_url or "",
                model=env.asr_model,
                ocr_first=env.asr_ocr_subs,
            ),
        ),
    )


def strip_secrets(app: AppSettings) -> dict:
    """Snapshot for a job manifest: everything except credentials (manifests are served)."""
    data = app.model_dump()
    for conn in data["connections"]:
        for field in SNAPSHOT_PRIVATE_FIELDS:
            conn[field] = ""
    return data


class SettingsStore:
    def __init__(self, env: Settings):
        self.env = env
        self.path: Path = env.private_dir / "settings.json"

    def load(self) -> AppSettings:
        if not self.path.exists():
            return seed_from_env(self.env)
        try:
            return AppSettings.model_validate_json(self.path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in err['loc']) or 'file'}: {err['msg']}"
                for err in exc.errors(include_input=False, include_url=False)
            )
            raise SettingsFileError(f"{self.path.name} không hợp lệ ({problems}) — lưu lại trong Cài đặt để ghi đè") from None

    def save(self, app: AppSettings) -> None:
        app.validate_references()
        with _LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(app.model_dump_json(indent=2))
            os.chmod(tmp, 0o600)
            tmp.replace(self.path)

    def public(self, app: AppSettings | None = None) -> dict:
        data = (app or self.load()).model_dump()
        for conn in data["connections"]:
            for field in SECRET_FIELDS:
                conn[field] = mask_secret(conn[field])
        return data

    def update(self, incoming: dict) -> AppSettings:
        """Save settings posted by the UI; a blank/masked secret keeps the stored one —
        except an API key whose connection now points at another service or address
        (same rule as ``rehydrate``: a key never travels to a different endpoint)."""
        app = AppSettings.model_validate(incoming)
        try:
            current = self.load()
        except SettingsFileError:  # unreadable file: the posted settings replace it
            current = seed_from_env(self.env)
        for conn in app.connections:
            old = current.connection(conn.id)
            moved = old is not None and (old.type, old.base_url) != (conn.type, conn.base_url)
            for field in SECRET_FIELDS:
                if _is_placeholder(getattr(conn, field)):
                    keep = old is not None and not (field == "api_key" and moved)
                    setattr(conn, field, getattr(old, field) if keep else "")
        self.save(app)
        return app

    def rehydrate(self, snapshot: dict) -> AppSettings:
        """A job's snapshot with the CURRENT credentials of its connections filled back in.

        Only when the connection still points at the same kind of service + host: a
        rotated key follows the job, but a key never travels to a different endpoint."""
        app = AppSettings.model_validate(snapshot)
        current = self.load()
        for conn in app.connections:
            live = current.connection(conn.id)
            if live is None or (live.type, live.base_url) != (conn.type, conn.base_url):
                continue
            for field in SNAPSHOT_PRIVATE_FIELDS:
                if not getattr(conn, field):
                    setattr(conn, field, getattr(live, field))
        return app


__all__ = ["MASK", "SettingsFileError", "SettingsStore", "mask_secret", "seed_from_env", "strip_secrets"]
