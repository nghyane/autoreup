import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def _node_major() -> int:
    out = subprocess.run([NODE, "--version"], capture_output=True, text=True, check=True).stdout
    return int(out.strip().lstrip("v").split(".")[0])


@unittest.skipUnless(NODE and _node_major() >= 23, "cần Node >= 23 (tự strip type TypeScript)")
class SettingsUiModelTest(unittest.TestCase):
    """The settings modal's settings <-> view-model helpers (frontend/src/settings/model.ts)."""

    def test_node_suite_passes(self):
        result = subprocess.run(
            [NODE, "--test", "--no-warnings", str(ROOT / "frontend/tests/settings-model.test.mjs")],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
