import os
import shlex
import subprocess
import logging
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional


# --- Audit logging setup ---

_AUDIT_LOGGER_NAME = "command_audit_logger"
_AUDIT_ENV_VAR = "COMMAND_AUDIT_LOG"
_DEFAULT_AUDIT_LOG_PATH = "command_audit.log.jsonl"


def _get_audit_log_path() -> str:
    path = os.environ.get(_AUDIT_ENV_VAR)
    if path and path.strip():
        return path
    return os.path.join(os.getcwd(), _DEFAULT_AUDIT_LOG_PATH)


def _get_audit_logger() -> logging.Logger:
    logger = logging.getLogger(_AUDIT_LOGGER_NAME)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(_get_audit_log_path(), encoding="utf-8")
    # We log pre-formatted JSON strings; keep formatter minimal
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(value: Optional[str], limit: int = 4096) -> Optional[str]:
    if value is None:
        return None
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _audit_log(event: dict) -> None:
    logger = _get_audit_logger()
    record = {"ts": _now_iso(), **event}
    logger.info(json.dumps(record, ensure_ascii=False, separators=(",", ":")))


# --- Core logic ---


def validate_command(shell_cmd: str, safe_subcommands: List[str]) -> List[str]:
    """
    Validate the provided shell command string against the list of allowed subcommands.
    Returns the tokenized command if valid; otherwise raises ValueError.

    This function is pure: it performs no I/O and depends only on its inputs.
    """
    if not isinstance(shell_cmd, str) or not shell_cmd.strip():
        raise ValueError("Command must be a non-empty string.")
    if not isinstance(safe_subcommands, list) or not all(isinstance(s, str) for s in safe_subcommands):
        raise ValueError("safe_subcommands must be a list of strings.")

    tokens = shlex.split(shell_cmd)
    if not tokens:
        raise ValueError("Command is empty after parsing.")
    if len(tokens) < 2:
        raise ValueError("No subcommand provided; command is not allowed.")

    # Validate the first subcommand token is permitted (maintains original behavior).
    subcommand = tokens[1]
    if subcommand not in set(safe_subcommands):
        raise ValueError(f"Subcommand '{subcommand}' is not allowed.")

    return tokens


@dataclass
class ExecResult:
    args: List[str]
    returncode: int
    stdout: str
    stderr: str
    error: Optional[str] = None  # Non-process exceptions (e.g., FileNotFoundError)


def execute_command(tokens: List[str]) -> ExecResult:
    """
    Execute the tokenized command without invoking a shell and return a structured result.
    """
    try:
        proc = subprocess.run(tokens, capture_output=True, text=True, shell=False)
        return ExecResult(
            args=tokens,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            error=None,
        )
    except FileNotFoundError:
        return ExecResult(
            args=tokens,
            returncode=127,  # Conventional code for command not found
            stdout="",
            stderr="",
            error=f"Command not found: {tokens[0]}",
        )


def validate_and_execute_command(shell_cmd: str, safe_subcommands: List[str]) -> str:
    """
    Execute a shell command only if it passes through specific filters,
    ensuring only certain subcommands are allowed.

    Args:
        shell_cmd: The complete command string to execute.
        safe_subcommands: A list of permitted subcommands (e.g., ["status", "pull"]).

    Returns:
        The stdout of the successful command, or an error message string if the command
        fails to execute successfully.

    Raises:
        ValueError: If the command is empty or if the subcommand is not permitted.
    """
    # Log the attempt before validation
    _audit_log(
        {
            "event": "attempt",
            "command": shell_cmd,
        }
    )

    try:
        tokens = validate_command(shell_cmd, safe_subcommands)
    except ValueError as e:
        _audit_log(
            {
                "event": "validation_error",
                "command": shell_cmd,
                "success": False,
                "error": str(e),
            }
        )
        raise

    result = execute_command(tokens)
    success = result.error is None and result.returncode == 0

    _audit_log(
        {
            "event": "execution",
            "command": shell_cmd,
            "tokens": result.args,
            "success": success,
            "returncode": result.returncode,
            "error": result.error or (_truncate(result.stderr) if result.stderr and not success else None),
        }
    )

    if success:
        return result.stdout
    return result.stderr or result.error or f"Command failed with exit code {result.returncode}."
