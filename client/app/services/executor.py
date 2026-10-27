"""Permission-gated local action execution."""

import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# Only read-only command forms are allowed; callers cannot invoke shells/scripts.
SAFE_COMMANDS = {
    ("git", "status"),
    ("git", "diff"),
    ("git", "log"),
    ("python", "--version"),
    ("python3", "--version"),
}


class ActionExecutor:
    """Execute supported Brain actions within explicit local safety limits."""

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        action_type = action.get("action", "none")
        payload = action.get("payload", {})
        if not isinstance(payload, dict):
            return {"status": "error", "message": "Action payload must be an object."}

        if action_type == "file_edit":
            return self._file_edit(payload)
        if action_type == "terminal_command":
            return self._terminal_command(payload)
        if action_type in {"none", "think"}:
            logger.info("Ignoring non-system action: %s", action_type)
            return {"status": "ignored", "action": action_type}
        return {"status": "error", "message": f"Unsupported action: {action_type}"}

    def _file_edit(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not settings.allow_file_edits:
            return {"status": "denied", "message": "File edits are disabled by client settings."}
        path_value, content = payload.get("path"), payload.get("content")
        if not isinstance(path_value, str) or not isinstance(content, str):
            return {"status": "error", "message": "file_edit requires string path and content."}

        root = settings.workspace_dir.resolve()
        target = (root / path_value).resolve()
        if not target.is_relative_to(root):
            return {"status": "denied", "message": "File path must stay within the configured workspace."}
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {"status": "success", "path": str(target)}
        except OSError as exc:
            logger.exception("Could not write requested file")
            return {"status": "error", "message": str(exc)}

    def _terminal_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not settings.allow_terminal_commands:
            return {"status": "denied", "message": "Terminal commands are disabled by client settings."}
        command = payload.get("command")
        if not isinstance(command, str):
            return {"status": "error", "message": "terminal_command requires a command string."}
        try:
            args = shlex.split(command, posix=False)
        except ValueError as exc:
            return {"status": "error", "message": f"Invalid command: {exc}"}
        normalized_args = [Path(args[0]).name.lower(), *args[1:]] if args else []
        if not normalized_args or not any(
            tuple(normalized_args[: len(allowed)]) == allowed
            for allowed in SAFE_COMMANDS
        ):
            return {"status": "denied", "message": "Command is not in the client safe-command allowlist."}
        try:
            completed = subprocess.run(
                args,
                cwd=settings.workspace_dir,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                shell=False,
            )
            return {
                "status": "success" if completed.returncode == 0 else "error",
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.exception("Terminal command failed")
            return {"status": "error", "message": str(exc)}
