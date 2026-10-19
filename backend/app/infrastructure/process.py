import os
import signal
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path


class ProcessError(RuntimeError):
    pass


def kill_tree(process: subprocess.Popen) -> None:
    """Kill a process started with ``start_new_session=True`` and everything it spawned."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


def run_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
    cancel: threading.Event | None = None,
) -> list[str]:
    """Run ``command`` and return its output lines; ``on_line`` sees each line as it
    arrives. Setting ``cancel`` kills the command and raises ProcessError."""
    logs: list[str] = []
    with subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=cancel is not None,
    ) as process:
        assert process.stdout is not None
        if cancel is not None:
            def watch() -> None:
                while process.poll() is None:
                    if cancel.wait(0.5):
                        kill_tree(process)
                        return

            threading.Thread(target=watch, daemon=True).start()
        for line in process.stdout:
            logs.append(line.rstrip())
            if on_line is not None:
                on_line(logs[-1])
        code = process.wait()
    if cancel is not None and cancel.is_set():
        raise ProcessError(f"canceled: {' '.join(command[:2])}")
    if code != 0:
        tail = "\n".join(logs[-30:])
        raise ProcessError(f"Command failed with exit code {code}: {' '.join(command)}\n{tail}")
    return logs
