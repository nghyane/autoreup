"""User settings schema — connections (where a capability lives) + tasks (who does what).

A *connection* is one configured provider (a CLI agent, an LLM API, a TTS service);
a *task* binds a pipeline step to a connection + model. Capabilities decide which
connections a task may use, so the UI only offers ones that can actually do the job.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

ConnectionType = Literal["agy", "claude_code", "codex", "command", "openai_compat", "gemini", "elevenlabs", "capcut", "edge", "http_tts", "vieneu"]

# What each connection type can do. ASR backends (local mlx / Cloudflare worker) stay
# their own task fields — they are not connections.
CAPABILITIES: dict[str, set[str]] = {
    "chat": {"agy", "claude_code", "codex", "command", "openai_compat", "gemini"},
    "vision": {"openai_compat", "gemini"},
    "tts": {"capcut", "http_tts", "edge", "openai_compat", "gemini", "elevenlabs", "vieneu"},
}
# Subscription CLI agents: the binary run when a connection leaves ``bin`` blank.
CLI_BINS: dict[str, str] = {"agy": "agy", "claude_code": "claude", "codex": "codex"}
SECRET_FIELDS = ("api_key", "password")
# Never written into a (served) job manifest; refilled from the store on resume.
SNAPSHOT_PRIVATE_FIELDS = (*SECRET_FIELDS, "email")
CHAT_TASKS = ("translate", "glossary", "normalize", "repair", "naming")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


class Connection(BaseModel):
    id: str
    label: str = ""
    type: ConnectionType
    base_url: str = ""   # API root; capcut: legacy HTTP TTS fallback server
    api_key: str = ""    # secret
    bin: str = ""        # CLI agent binary (blank = CLI_BINS default)
    home: str = ""       # HOME override for CLI agents (account isolation)
    command: str = ""    # provider=command template with {prompt}
    email: str = ""      # capcut auto-login
    password: str = ""   # secret


class ChatTask(BaseModel):
    connection: str = ""
    model: str = ""
    effort: str = ""


class OcrTask(BaseModel):
    recognizer: Literal["vision", "llm"] = "vision"
    connection: str = ""
    model: str = ""


class TtsTask(BaseModel):
    enabled: bool = False
    connection: str = ""
    model: str = ""
    voice: str = ""
    synth_rate: float = Field(1.0, gt=0, le=4)


class AsrTask(BaseModel):
    backend: Literal["auto", "qwen", "local", "worker"] = "auto"
    worker_url: str = ""
    model: str = ""
    ocr_first: bool = True


class Tasks(BaseModel):
    translate: ChatTask = Field(default_factory=ChatTask)
    glossary: ChatTask = Field(default_factory=ChatTask)
    normalize: ChatTask = Field(default_factory=ChatTask)
    repair: ChatTask = Field(default_factory=ChatTask)
    naming: ChatTask = Field(default_factory=ChatTask)
    ocr: OcrTask = Field(default_factory=OcrTask)
    tts: TtsTask = Field(default_factory=TtsTask)
    asr: AsrTask = Field(default_factory=AsrTask)


class AppSettings(BaseModel):
    version: int = 1
    connections: list[Connection] = Field(default_factory=list)
    tasks: Tasks = Field(default_factory=Tasks)

    def connection(self, connection_id: str) -> Connection | None:
        return next((c for c in self.connections if c.id == connection_id), None)

    def validate_references(self) -> None:
        """Raise ValueError when ids clash or a task points at an unusable connection."""
        seen: set[str] = set()
        for conn in self.connections:
            if not _ID_RE.match(conn.id):
                raise ValueError(f"id kết nối không hợp lệ: {conn.id!r}")
            if conn.id in seen:
                raise ValueError(f"id kết nối bị trùng: {conn.id}")
            seen.add(conn.id)
        for name in CHAT_TASKS:
            self._require(name, getattr(self.tasks, name).connection, "chat")
        if self.tasks.ocr.recognizer == "llm":
            self._require("ocr", self.tasks.ocr.connection, "vision")
        if self.tasks.tts.enabled or self.tasks.tts.connection:
            self._require("tts", self.tasks.tts.connection, "tts")

    def _require(self, task: str, connection_id: str, capability: str) -> None:
        conn = self.connection(connection_id)
        if conn is None:
            raise ValueError(f"tác vụ {task}: không có kết nối {connection_id!r}")
        if conn.type not in CAPABILITIES[capability]:
            raise ValueError(f"tác vụ {task}: kết nối {conn.id} ({conn.type}) không hỗ trợ {capability}")


__all__ = [
    "CAPABILITIES",
    "CHAT_TASKS",
    "CLI_BINS",
    "SECRET_FIELDS",
    "SNAPSHOT_PRIVATE_FIELDS",
    "AppSettings",
    "AsrTask",
    "ChatTask",
    "Connection",
    "ConnectionType",
    "OcrTask",
    "Tasks",
    "TtsTask",
]
