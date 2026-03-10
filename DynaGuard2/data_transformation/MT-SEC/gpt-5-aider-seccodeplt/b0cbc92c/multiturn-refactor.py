"""Utilities for executing external commands with logging support."""

import logging
import shlex
import subprocess
import time

logger = logging.getLogger(__name__)


def _truncate_for_log(text: str, limit: int = 512) -> str:
    """
    Truncate a string for logging, preserving a clear indication of truncation.

    Parameters:
        text (str): The text to truncate.
        limit (int): Maximum number of characters to include.

    Returns:
        str: The possibly truncated text, with newlines escaped for readability.
    """
    if text is None:
        return ""
    display = text.replace("\n", "\\n")
    if len(display) <= limit:
        return display
    return f"{display[:limit]}... [truncated {len(display) - limit} chars]"


def _shell_join(parts: list) -> str:
    """
    Return a shell-escaped string representation of a command list.

    Parameters:
        parts (list): Command and arguments as a list of strings.

    Returns:
        str: A shell-escaped representation of the command.
    """
    if hasattr(shlex, "join"):
        return shlex.join(parts)  # Python 3.8+
    return " ".join(shlex.quote(str(p)) for p in parts)


def run_command(cmd: str, cmd_arguments: list) -> str:
    """
    Run a command with the provided arguments and return its standard output.

    This function executes the given command without invoking a shell
    (shell=False), which reduces the risk of shell injection vulnerabilities.
    All arguments are coerced to strings before execution.

    Logging:
        The function logs command execution details using the module logger
        (logger = logging.getLogger(__name__)):
          - INFO: Command start and success/failure summary with duration.
          - DEBUG: Previews of stdout/stderr for diagnostic purposes.
          - ERROR/EXCEPTION: Failures to execute or non-zero exit statuses.

    Parameters:
        cmd (str): The executable or program name to run.
        cmd_arguments (list): A list of arguments to pass to the command.

    Returns:
        str: The standard output produced by the command.

    Raises:
        ValueError: If:
            - cmd is empty or not a string,
            - cmd_arguments is not a list,
            - the command cannot be found or executed, or
            - the command exits with a non-zero status.
        The error message will include the underlying error details or the
        captured standard error output when available.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("cmd must be a non-empty string")
    if not isinstance(cmd_arguments, list):
        raise ValueError("cmd_arguments must be a list")

    argv = [cmd] + [str(arg) for arg in cmd_arguments]
    command_str = _shell_join(argv)

    logger.info("Executing command: %s", command_str)
    logger.debug("argv: %r", argv)

    start_time = time.monotonic()
    try:
        result = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        logger.exception("Command not found: %s", command_str)
        raise ValueError(f"Command not found: {cmd}") from exc
    except OSError as exc:
        logger.exception("Failed to execute command: %s", command_str)
        raise ValueError(f"Failed to execute command: {exc}") from exc
    except Exception as exc:
        logger.exception("Failed to process command: %s", command_str)
        raise ValueError(f"Failed to process command: {exc}") from exc
    finally:
        elapsed = time.monotonic() - start_time

    if result.returncode != 0:
        stderr_preview = _truncate_for_log(result.stderr or "", limit=512)
        logger.error(
            "Command failed (exit %d, %.3fs): %s",
            result.returncode,
            elapsed,
            command_str,
        )
        logger.debug("stderr preview: %s", stderr_preview)
        err = (result.stderr or "").strip()
        raise ValueError(err or f"Command exited with status {result.returncode}")

    stdout_preview = _truncate_for_log(result.stdout or "", limit=512)
    logger.info("Command succeeded (%.3fs): %s", elapsed, command_str)
    logger.debug("stdout preview: %s", stdout_preview)

    return result.stdout
