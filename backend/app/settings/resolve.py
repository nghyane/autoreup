"""Turn settings into the runtime objects a stage needs (``resolve(task)``).

Stages never read provider/model from env directly any more: they ask for the task's
binding here. Infra knobs (timeouts, retries, paths) still come from env ``Settings``.
"""

from __future__ import annotations

from backend.app.agents import AgentConfig
from backend.app.core.config import Settings

from .schema import CAPABILITIES, CLI_BINS, AppSettings, AsrTask, Connection, OcrTask, TtsTask
from .store import SettingsStore


def chat_agent(app: AppSettings, task: str, env: Settings) -> AgentConfig:
    """AgentConfig for one chat task (translate, glossary, normalize, repair, naming)."""
    binding = getattr(app.tasks, task)
    conn = app.connection(binding.connection)
    if conn is None:
        raise ValueError(f"tác vụ {task}: không có kết nối {binding.connection!r}")
    if conn.type not in CAPABILITIES["chat"]:
        # The agy workspace runner treats unknown providers as agy: fail loudly instead
        # of spending agy quota on a model id meant for another service.
        raise ValueError(f"tác vụ {task}: kết nối {conn.id} ({conn.type}) không chạy được tác vụ chat")
    return agent_for(conn, binding.model, env, effort=binding.effort)


def agent_for(conn: Connection, model: str, env: Settings, effort: str = "") -> AgentConfig:
    return AgentConfig(
        provider=conn.type,
        model=model,
        effort=effort,
        bin=conn.bin or CLI_BINS.get(conn.type, "agy"),
        home=conn.home or None,
        command=conn.command,
        timeout=env.agent_timeout,
        retries=env.agent_retries,
        base_url=conn.base_url,
        api_key=conn.api_key,
    )


def ocr_task(app: AppSettings) -> OcrTask:
    return app.tasks.ocr


def ocr_agent(app: AppSettings, env: Settings) -> AgentConfig:
    """AgentConfig for ``tasks.ocr.recognizer == "llm"`` — a vision-capable API connection."""
    binding = app.tasks.ocr
    conn = app.connection(binding.connection)
    if conn is None:
        raise ValueError(f"tác vụ ocr: không có kết nối {binding.connection!r}")
    if conn.type not in CAPABILITIES["vision"]:
        raise ValueError(f"tác vụ ocr: kết nối {conn.id} ({conn.type}) không đọc được ảnh")
    return agent_for(conn, binding.model, env)


def asr_task(app: AppSettings) -> AsrTask:
    return app.tasks.asr


def tts_binding(app: AppSettings) -> tuple[TtsTask, Connection | None]:
    task = app.tasks.tts
    return task, app.connection(task.connection)


def app_settings_for(context) -> AppSettings:
    """The settings a pipeline run uses: its job snapshot (secrets re-filled from the
    store), or the current store for jobs created before snapshots existed."""
    store = SettingsStore(context.settings)
    snapshot = getattr(context.config, "settings_snapshot", None)
    if snapshot:
        return store.rehydrate(snapshot)
    return store.load()


__all__ = ["agent_for", "app_settings_for", "asr_task", "chat_agent", "ocr_agent", "ocr_task", "tts_binding"]
