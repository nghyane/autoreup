"""Settings store: JSON overrides env, seeded from env when absent, secrets never leak."""

import json
import tempfile
import unittest
from pathlib import Path

from backend.app.agents import AgentConfig
from backend.app.core.config import Settings
from backend.app.settings import (
    AppSettings,
    SettingsStore,
    app_settings_for,
    asr_task,
    chat_agent,
    seed_from_env,
    strip_secrets,
)


def _env(root: Path, **overrides) -> Settings:
    base = dict(root_dir=root, _env_file=None)
    base.update(overrides)
    settings = Settings(**base)
    settings.ensure_directories()
    return settings


def _legacy_agent(settings: Settings, model: str) -> AgentConfig:
    """The AgentConfig the pipeline built straight from env before the settings store."""
    return AgentConfig(
        provider=settings.agent_provider,
        bin=settings.agent_bin,
        home=settings.agent_home,
        command=settings.agent_command,
        model=model,
        effort=settings.agent_effort,
        timeout=settings.agent_timeout,
        retries=settings.agent_retries,
    )


class SeedParityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_file_resolves_exactly_like_env(self):
        env = _env(self.root, agent_home="~/.agy-accounts/a2", agent_effort="low")
        app = SettingsStore(env).load()
        self.assertFalse((env.private_dir / "settings.json").exists())  # load never writes
        for task in ("translate", "glossary", "normalize", "naming"):
            self.assertEqual(chat_agent(app, task, env), _legacy_agent(env, env.translate_model), task)
        self.assertEqual(chat_agent(app, "repair", env), _legacy_agent(env, env.polish_model))

    def test_seed_command_provider(self):
        env = _env(self.root, agent_provider="command", agent_command="acpx claude exec {prompt}")
        app = seed_from_env(env)
        agent = chat_agent(app, "translate", env)
        self.assertEqual(agent.provider, "command")
        self.assertEqual(agent.command, "acpx claude exec {prompt}")
        self.assertEqual(agent, _legacy_agent(env, env.translate_model))

    def test_seed_claude_and_codex_providers_use_their_own_binary(self):
        for provider, conn_type, binary in (("claude", "claude_code", "claude"), ("codex", "codex", "codex")):
            env = _env(self.root, agent_provider=provider)
            agent = chat_agent(seed_from_env(env), "translate", env)
            self.assertEqual((agent.provider, agent.bin), (conn_type, binary), provider)

    def test_seed_tts_native_capcut_keeps_http_fallback(self):
        env = _env(self.root, tts_enabled=True, tts_native=True, tts_server_url="http://127.0.0.1:8080",
                   capcut_email="a@b.c", capcut_password="pw")
        app = seed_from_env(env)
        conn = app.connection(app.tasks.tts.connection)
        self.assertEqual(conn.type, "capcut")
        self.assertEqual(conn.base_url, "http://127.0.0.1:8080")
        self.assertEqual((conn.email, conn.password), ("a@b.c", "pw"))
        self.assertTrue(app.tasks.tts.enabled)
        self.assertEqual(app.tasks.tts.voice, env.tts_voice)

    def test_seed_tts_http_when_not_native(self):
        env = _env(self.root, tts_native=False, tts_server_url="http://tts:8080")
        app = seed_from_env(env)
        conn = app.connection(app.tasks.tts.connection)
        self.assertEqual((conn.type, conn.base_url), ("http_tts", "http://tts:8080"))
        self.assertFalse(app.tasks.tts.enabled)

    def test_seed_asr_and_ocr(self):
        env = _env(self.root, asr_backend="worker", asr_worker_url="https://w.example", asr_ocr_subs=False)
        app = seed_from_env(env)
        asr = asr_task(app)
        self.assertEqual((asr.backend, asr.worker_url, asr.model, asr.ocr_first),
                         ("worker", "https://w.example", env.asr_model, False))
        self.assertEqual(app.tasks.ocr.recognizer, "vision")


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = _env(Path(self.tmp.name))
        self.store = SettingsStore(self.env)

    def tearDown(self):
        self.tmp.cleanup()

    def _with_api_connection(self, key: str = "sk-secret-123456") -> AppSettings:
        app = self.store.load()
        data = app.model_dump()
        data["connections"].append({"id": "oa", "label": "OpenAI", "type": "openai_compat",
                                    "base_url": "https://api.example/v1", "api_key": key})
        return AppSettings.model_validate(data)

    def test_roundtrip_and_file_is_private(self):
        app = self._with_api_connection()
        self.store.save(app)
        self.assertEqual(self.store.load(), app)
        mode = self.store.path.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_public_masks_secrets(self):
        self.store.save(self._with_api_connection())
        public = self.store.public()
        conn = next(c for c in public["connections"] if c["id"] == "oa")
        self.assertEqual(conn["api_key"], "••••3456")
        self.assertNotIn("sk-secret-123456", json.dumps(public))

    def test_update_keeps_secret_when_blank_or_masked_and_replaces_when_new(self):
        self.store.save(self._with_api_connection())
        incoming = self.store.public()
        self.store.update(incoming)  # masked value posted back
        self.assertEqual(self.store.load().connection("oa").api_key, "sk-secret-123456")
        for conn in incoming["connections"]:
            if conn["id"] == "oa":
                conn["api_key"] = ""
        self.store.update(incoming)
        self.assertEqual(self.store.load().connection("oa").api_key, "sk-secret-123456")
        for conn in incoming["connections"]:
            if conn["id"] == "oa":
                conn["api_key"] = "sk-new"
        self.store.update(incoming)
        self.assertEqual(self.store.load().connection("oa").api_key, "sk-new")

    def test_update_never_carries_a_stored_key_to_another_address(self):
        self.store.save(self._with_api_connection())
        incoming = self.store.public()
        for conn in incoming["connections"]:
            if conn["id"] == "oa":
                conn["base_url"] = "https://elsewhere.example/v1"  # key left masked
        self.store.update(incoming)
        self.assertEqual(self.store.load().connection("oa").api_key, "")

    def test_validation_rejects_bad_references(self):
        data = self.store.load().model_dump()
        data["tasks"]["translate"]["connection"] = "missing"
        with self.assertRaises(ValueError):
            self.store.update(data)

        data = self._with_api_connection().model_dump()
        data["tasks"]["tts"]["connection"] = data["tasks"]["translate"]["connection"]  # agy cannot speak
        with self.assertRaises(ValueError):
            self.store.update(data)

        data = self._with_api_connection().model_dump()
        data["connections"].append(dict(data["connections"][-1]))
        with self.assertRaises(ValueError):
            self.store.update(data)

        data = self.store.load().model_dump()
        data["tasks"]["ocr"] = {"recognizer": "llm", "connection": data["tasks"]["translate"]["connection"], "model": "m"}
        with self.assertRaises(ValueError):  # agy has no vision
            self.store.update(data)

    def test_snapshot_strips_secrets_and_rehydrates_current_ones(self):
        self.store.save(self._with_api_connection("sk-old-000000"))
        snapshot = strip_secrets(self.store.load())
        self.assertNotIn("sk-old-000000", json.dumps(snapshot))
        app = self.store.load()
        data = app.model_dump()
        data["connections"][-1]["api_key"] = "sk-rotated-1"
        self.store.save(AppSettings.model_validate(data))
        rehydrated = self.store.rehydrate(snapshot)
        self.assertEqual(rehydrated.connection("oa").api_key, "sk-rotated-1")

    def test_app_settings_for_context_prefers_snapshot(self):
        from types import SimpleNamespace

        app = self.store.load()
        data = app.model_dump()
        data["tasks"]["translate"]["model"] = "snap-model"
        snapshot = strip_secrets(AppSettings.model_validate(data))
        context = SimpleNamespace(settings=self.env, config=SimpleNamespace(settings_snapshot=snapshot))
        self.assertEqual(app_settings_for(context).tasks.translate.model, "snap-model")
        legacy = SimpleNamespace(settings=self.env, config=SimpleNamespace(settings_snapshot=None))
        self.assertEqual(app_settings_for(legacy).tasks.translate.model, self.env.translate_model)

    def test_rehydrate_never_sends_a_key_to_a_different_host(self):
        self.store.save(self._with_api_connection("sk-old-000000"))
        snapshot = strip_secrets(self.store.load())
        data = self.store.load().model_dump()
        data["connections"][-1].update(base_url="https://other.example/v1", api_key="sk-other-1")
        self.store.save(AppSettings.model_validate(data))
        self.assertEqual(self.store.rehydrate(snapshot).connection("oa").api_key, "")

    def test_chat_task_on_api_connection_resolves_to_the_api_provider(self):
        data = self._with_api_connection().model_dump()
        data["tasks"]["translate"].update(connection="oa", model="gpt-x", effort="low")
        agent = chat_agent(AppSettings.model_validate(data), "translate", self.env)
        self.assertEqual((agent.provider, agent.model, agent.effort), ("openai_compat", "gpt-x", "low"))
        self.assertEqual((agent.base_url, agent.api_key), ("https://api.example/v1", "sk-secret-123456"))

    def test_chat_task_on_non_chat_connection_fails_loudly_instead_of_running_agy(self):
        data = self.store.load().model_dump()
        data["connections"].append({"id": "edge", "type": "edge"})
        data["tasks"]["translate"]["connection"] = "edge"
        app = AppSettings.model_validate(data)
        with self.assertRaises(ValueError):
            chat_agent(app, "translate", self.env)

    def test_corrupt_file_reports_without_echoing_content_and_can_be_overwritten(self):
        from backend.app.settings import SettingsFileError

        self.store.path.write_text('{"connections": [{"id": "x", "api_key": "sk-leak-me"', encoding="utf-8")
        with self.assertRaises(SettingsFileError) as caught:
            self.store.load()
        self.assertNotIn("sk-leak-me", str(caught.exception))
        self.store.update(seed_from_env(self.env).model_dump())
        self.assertEqual(self.store.load(), seed_from_env(self.env))

    def test_synth_rate_must_be_positive(self):
        data = self.store.load().model_dump()
        data["tasks"]["tts"]["synth_rate"] = 0
        with self.assertRaises(ValueError):
            self.store.update(data)


class JobSnapshotTest(unittest.TestCase):
    def test_created_job_carries_secret_free_snapshot(self):
        from backend.app.domain.jobs import JobInput
        from backend.app.services.job_service import JobService

        with tempfile.TemporaryDirectory() as tmp:
            env = _env(Path(tmp), tts_native=True, capcut_email="me@x.y", capcut_password="hunter2-password")
            service = JobService(env)
            manifest = service.create(JobInput(url="https://example.com/v"))
            snapshot = manifest.config.settings_snapshot
            self.assertIsNotNone(snapshot)
            raw = service.manifest_path(manifest.job_id).read_text(encoding="utf-8")
            self.assertNotIn("hunter2-password", raw)
            self.assertNotIn("me@x.y", raw)
            self.assertEqual(snapshot["tasks"]["translate"]["model"], env.translate_model)


class TtsEngineTest(unittest.TestCase):
    """CapCut synthesizes natively, falls back to its HTTP base_url, fails only when neither works."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = _env(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_capcut_native_then_http_fallback_then_error(self):
        from unittest.mock import patch

        import backend.app.pipeline.production as prod
        from backend.app.settings import Connection, TtsTask

        conn = Connection(id="capcut", type="capcut", base_url="http://tts:8080")
        client = object()
        with patch.object(prod, "_build_capcut_client", return_value=client):
            engine = prod._tts_engine(self.env, conn, TtsTask(voice="v"))
            self.assertEqual((engine.client, engine.cache_id, engine.voice), (client, "capcut-native", "v"))
        with patch.object(prod, "_build_capcut_client", return_value=None):
            self.assertEqual(prod._tts_engine(self.env, conn, TtsTask(voice="v")).cache_id, "http://tts:8080")
            with self.assertRaises(RuntimeError):
                prod._tts_engine(self.env, Connection(id="capcut", type="capcut"), TtsTask(voice="v"))

    def test_http_tts_needs_base_url_and_incomplete_engines_fail_loudly(self):
        import backend.app.pipeline.production as prod
        from backend.app.settings import Connection, TtsTask

        engine = prod._tts_engine(self.env, Connection(id="t", type="http_tts", base_url="http://x"), TtsTask(voice="v"))
        self.assertEqual((engine.cache_id, engine.voice), ("http://x", "v"))
        for conn in (Connection(id="t", type="http_tts"), Connection(id="o", type="openai_compat"), None):
            with self.assertRaises(RuntimeError):
                prod._tts_engine(self.env, conn, TtsTask(voice="v"))

    def test_voiceover_reads_enabled_flag_from_job_snapshot(self):
        from types import SimpleNamespace

        from backend.app.pipeline.production import VoiceoverStage

        data = SettingsStore(self.env).load().model_dump()
        data["tasks"]["tts"]["enabled"] = False
        context = SimpleNamespace(settings=self.env, config=SimpleNamespace(settings_snapshot=strip_secrets(AppSettings.model_validate(data))),
                                  artifacts={"speech_rows": []}, job_id="j")
        self.assertEqual(VoiceoverStage().run(context).metrics, {"enabled": False})


if __name__ == "__main__":
    unittest.main()
