"""/settings API: masked reads, secret-preserving writes, validation, connection probes."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.app.api.v1.endpoints.settings as settings_endpoint
from backend.app.agents import AgentError
from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.settings import SettingsStore


class SettingsApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        env = Settings(root_dir=Path(self.tmp.name), _env_file=None, agent_bin="agy-binary-that-does-not-exist")
        env.ensure_directories()
        self.store = SettingsStore(env)
        self.patcher = patch.object(settings_endpoint, "get_settings_store", return_value=self.store)
        self.patcher.start()
        self.client = TestClient(create_app())

    def tearDown(self):
        self.patcher.stop()
        self.tmp.cleanup()

    def test_get_put_roundtrip_never_returns_secret(self):
        data = self.client.get("/api/v1/settings").json()
        self.assertIn("connections", data)
        self.assertIn("translate", data["tasks"])
        data["connections"].append({"id": "gm", "label": "Gemini", "type": "gemini", "api_key": "AIza-very-secret-key"})
        saved = self.client.put("/api/v1/settings", json=data)
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertNotIn("AIza-very-secret-key", saved.text)
        # posting the masked value back keeps the stored secret
        again = self.client.put("/api/v1/settings", json=saved.json())
        self.assertEqual(again.status_code, 200)
        self.assertEqual(self.store.load().connection("gm").api_key, "AIza-very-secret-key")
        self.assertNotIn("AIza-very-secret-key", self.client.get("/api/v1/settings").text)

    def test_put_invalid_reference_is_400(self):
        data = self.client.get("/api/v1/settings").json()
        data["tasks"]["translate"]["connection"] = "nope"
        response = self.client.put("/api/v1/settings", json=data)
        self.assertEqual(response.status_code, 400)

    def test_probe_agy_connection_reports_missing_binary(self):
        data = self.client.get("/api/v1/settings").json()
        conn_id = data["tasks"]["translate"]["connection"]
        result = self.client.post(f"/api/v1/settings/connections/{conn_id}/test").json()
        self.assertFalse(result["ok"])
        self.assertIn("agy-binary-that-does-not-exist", result["message"])
        models = self.client.get(f"/api/v1/settings/connections/{conn_id}/models")
        self.assertEqual(models.status_code, 200)
        self.assertEqual(models.json(), {"models": []})

    def test_claude_and_codex_connections_probe_their_binary_and_suggest_models(self):
        data = self.client.get("/api/v1/settings").json()
        data["connections"] += [
            {"id": "cc", "label": "Claude Code", "type": "claude_code", "bin": "claude-binary-that-does-not-exist"},
            {"id": "cx", "label": "Codex", "type": "codex", "home": self.tmp.name},
        ]
        data["tasks"]["translate"] = {"connection": "cc", "model": "sonnet", "effort": "high"}
        self.assertEqual(self.client.put("/api/v1/settings", json=data).status_code, 200)
        probe = self.client.post("/api/v1/settings/connections/cc/test").json()
        self.assertFalse(probe["ok"])
        self.assertIn("claude-binary-that-does-not-exist", probe["message"])
        self.assertIn("sonnet", self.client.get("/api/v1/settings/connections/cc/models").json()["models"])
        # codex: suggestions come from the CLI's own model cache under that HOME
        self.assertEqual(self.client.get("/api/v1/settings/connections/cx/models").json(), {"models": []})
        cache = Path(self.tmp.name) / ".codex" / "models_cache.json"
        cache.parent.mkdir()
        cache.write_text('{"models": [{"slug": "gpt-5.5"}, {"slug": "codex-auto-review"}]}', encoding="utf-8")
        self.assertEqual(self.client.get("/api/v1/settings/connections/cx/models").json(), {"models": ["gpt-5.5"]})

    def test_corrupt_file_is_a_clear_error_and_put_repairs_it(self):
        fresh = self.client.get("/api/v1/settings").json()
        self.store.path.write_text("{not json", encoding="utf-8")
        broken = self.client.get("/api/v1/settings")
        self.assertEqual(broken.status_code, 500)
        self.assertIn("settings.json", broken.json()["detail"])
        self.assertEqual(self.client.put("/api/v1/settings", json=fresh).status_code, 200)
        self.assertEqual(self.client.get("/api/v1/settings").status_code, 200)

    def test_probe_bad_url_reports_instead_of_500(self):
        data = self.client.get("/api/v1/settings").json()
        data["connections"].append({"id": "t", "type": "http_tts", "base_url": "http://[::1"})
        self.assertEqual(self.client.put("/api/v1/settings", json=data).status_code, 200)
        result = self.client.post("/api/v1/settings/connections/t/test")
        self.assertEqual(result.status_code, 200)
        self.assertFalse(result.json()["ok"])

    def _add_api_connection(self) -> str:
        data = self.client.get("/api/v1/settings").json()
        data["connections"].append({"id": "oa", "type": "openai_compat", "base_url": "https://api.example/v1", "api_key": "sk-secret-123456"})
        self.assertEqual(self.client.put("/api/v1/settings", json=data).status_code, 200)
        return "oa"

    def test_api_connection_probe_and_models_use_the_adapter_with_the_stored_key(self):
        conn_id = self._add_api_connection()
        seen = []

        def fake_list(config):
            seen.append((config.provider, config.base_url, config.api_key))
            return ["gpt-a", "gpt-b"]

        with patch.object(settings_endpoint.api_llm, "list_models", fake_list):
            probe = self.client.post(f"/api/v1/settings/connections/{conn_id}/test").json()
            models = self.client.get(f"/api/v1/settings/connections/{conn_id}/models").json()
        self.assertTrue(probe["ok"])
        self.assertIn("2", probe["message"])
        self.assertEqual(models, {"models": ["gpt-a", "gpt-b"]})
        self.assertEqual(seen[0], ("openai_compat", "https://api.example/v1", "sk-secret-123456"))

    def test_api_connection_failure_is_reported_not_500(self):
        conn_id = self._add_api_connection()

        def failing(config):
            raise AgentError("openai_compat HTTP 401: Incorrect API key provided")

        with patch.object(settings_endpoint.api_llm, "list_models", failing):
            probe = self.client.post(f"/api/v1/settings/connections/{conn_id}/test")
            models = self.client.get(f"/api/v1/settings/connections/{conn_id}/models")
        self.assertEqual(probe.status_code, 200)
        self.assertFalse(probe.json()["ok"])
        self.assertIn("401", probe.json()["message"])
        self.assertEqual(models.status_code, 502)
        self.assertNotIn("sk-secret-123456", probe.text + models.text)

    def test_unknown_connection_is_404(self):
        self.assertEqual(self.client.post("/api/v1/settings/connections/zzz/test").status_code, 404)


if __name__ == "__main__":
    unittest.main()
