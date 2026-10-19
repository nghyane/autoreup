import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import asr_worker_client


class ASRWorkerClientContractTest(unittest.TestCase):
    def test_call_worker_sends_selected_model(self):
        payloads = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return b'{"segments": []}'

        def fake_urlopen(request, timeout):
            payloads.append(json.loads(request.data.decode()))
            return FakeResponse()

        with tempfile.NamedTemporaryFile(suffix=".m4a") as audio:
            audio.write(b"fake-audio-bytes")
            audio.flush()
            with patch.object(asr_worker_client, "urlopen", fake_urlopen):
                asr_worker_client.call_worker(
                    "http://worker",
                    Path(audio.name),
                    offset=12.0,
                    length=30.0,
                    language="zh",
                    model="@cf/openai/whisper",
                )

        self.assertEqual(payloads[0]["model"], "@cf/openai/whisper")


if __name__ == "__main__":
    unittest.main()


def test_local_backend_model_mapping_and_auto_selection():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts import asr_worker_client as client

    # Cloudflare slugs map onto the MLX repos (same weights, local runtime)
    assert client.local_repo("@cf/openai/whisper-large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"
    assert client.local_repo("@cf/openai/whisper") == "mlx-community/whisper-large-v3-mlx"
    assert client.local_repo("mlx-community/whisper-small-mlx") == "mlx-community/whisper-small-mlx"
    # auto prefers Qwen3-ASR, then mlx-whisper, then the worker — the first installed
    if importlib.util.find_spec("mlx_qwen3_asr"):
        expected = "qwen"
    else:
        expected = "local" if importlib.util.find_spec("mlx_whisper") else "worker"
    assert client.pick_backend("auto") == expected
    assert client.pick_backend("worker") == "worker"
    assert client.pick_backend("local") == "local"
    assert client.pick_backend("qwen") == "qwen"


def test_qwen_repo_ignores_the_whisper_model_setting():
    from scripts import asr_worker_client as client

    assert client.qwen_repo("@cf/openai/whisper-large-v3-turbo") == client.QWEN_DEFAULT_REPO
    assert client.qwen_repo("") == client.QWEN_DEFAULT_REPO
    assert client.qwen_repo("Qwen/Qwen3-ASR-0.6B") == "Qwen/Qwen3-ASR-0.6B"


def test_qwen_phrases_keep_aligned_times_and_unaligned_chunks():
    import pytest

    pytest.importorskip("mlx_qwen3_asr")
    from scripts import asr_worker_client as client

    text = "往后退，别靠近现场。西郊野坡发现无名男尸。"
    chars = [c for c in text if c not in "，。"]
    words = [{"text": c, "start": 1.0 + 0.2 * i, "end": 1.2 + 0.2 * i} for i, c in enumerate(chars)]
    chunks = [
        {"text": text, "start": 0.0, "end": 30.0},
        # the aligner returned nothing for this chunk: its text must still come through
        {"text": "ご視聴ありがとう", "start": 30.0, "end": 34.0},
    ]
    phrases = client.phrases_from_qwen(words, chunks, "Chinese", text, max_chars=18)

    # a sentence end splits the cue, and each cue spans exactly its aligned characters
    assert [p["text"] for p in phrases] == ["往后退，别靠近现场。", "西郊野坡发现无名男尸。", "ご視聴ありがとう"]
    assert (phrases[0]["start"], phrases[0]["end"]) == (words[0]["start"], words[7]["end"])
    assert (phrases[1]["start"], phrases[1]["end"]) == (words[8]["start"], words[-1]["end"])
    assert (phrases[2]["start"], phrases[2]["end"]) == (30.0, 34.0)
