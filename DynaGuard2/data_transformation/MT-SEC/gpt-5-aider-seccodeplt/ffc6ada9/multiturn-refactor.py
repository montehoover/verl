"""Utilities to execute external commands safely and capture output.

This module exposes a single public function: execute_command().

Logging:
    This module uses the standard logging library. To see logs, configure
    logging in your application, for example:

        import logging

        logging.basicConfig(level=logging.INFO)
        # or for more verbosity:
        logging.basicConfig(level=logging.DEBUG)

    Logs emitted:
        - INFO: When a command is about to be executed (shell-quoted form).
        - DEBUG: Tokenized command parts, parameters, and captured stdout/stderr.
        - ERROR: When a command is not found or fails.
        - EXCEPTION: For unexpected errors during execution.
"""

import logging
import shlex
import subprocess

__all__ = ["execute_command"]

logger = logging.getLogger(__name__)


def _split_base_command(base_command: str) -> list:
    """Split the base command into executable and subcommands using shlex.

    Args:
        base_command (str): The main command to execute. This may include
            subcommands (e.g., "git status").

    Returns:
        list: The tokenized base command parts.

    Raises:
        ValueError: If the base command is not a valid, non-empty string or
            cannot be parsed.
    """
    if not isinstance(base_command, str) or not base_command.strip():
        raise ValueError("base_command must be a non-empty string")

    try:
        return shlex.split(base_command)
    except ValueError as exc:
        raise ValueError(f"Invalid base_command: {exc}") from exc


def _normalize_parameters(parameters: list) -> list:
    """Validate and normalize command parameters.

    Args:
        parameters (list): A list of arguments to pass to the command.

    Returns:
        list: A list of stringified parameters.

    Raises:
        ValueError: If parameters is not a list or contains values that cannot
            be converted to strings.
    """
    if not isinstance(parameters, list):
        raise ValueError("parameters must be a list")

    try:
        return [str(p) for p in parameters]
    except Exception as exc:  # noqa: BLE001 - re-raise as ValueError for API
        raise ValueError(f"Invalid parameters: {exc}") from exc


def _format_command(parts: list) -> str:
    """Return a shell-quoted string representation of a command parts list."""
    return " ".join(shlex.quote(p) for p in parts)


def _truncate_for_log(text: str | None, limit: int = 2000) -> str:
    """Truncate text for logging to avoid excessively large log entries."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}... [truncated {omitted} chars]"


def execute_command(base_command: str, parameters: list) -> str:
    """Execute a system command with arguments and return its standard output.

    The command is executed without invoking a shell for safety. The provided
    base_command is tokenized (allowing values like "git status"), and the
    parameters are appended as additional arguments.

    Logging:
        - Logs the fully composed command (shell-quoted) at INFO level.
        - Logs tokenized parts and captured stdout/stderr at DEBUG level.
        - Logs errors and unexpected failures appropriately.

    Args:
        base_command (str): The main command to execute (e.g., "ls",
            "git status").
        parameters (list): A list of arguments to pass to the command. All
            values will be converted to strings.

    Returns:
        str: The command's standard output (stdout) as a string.

    Raises:
        ValueError: If inputs are invalid, the command is not found, or the
            command exits with a non-zero status.

    Example:
        >>> execute_command("echo", ["hello", "world"])
        'hello world\\n'
    """
    cmd_parts = _split_base_command(base_command)
    param_parts = _normalize_parameters(parameters)
    full_cmd = cmd_parts + param_parts
    cmd_str = _format_command(full_cmd)

    logger.debug("Base command tokens: %s", cmd_parts)
    logger.debug("Parameter parts: %s", param_parts)
    logger.info("Executing command: %s", cmd_str)

    try:
        completed = subprocess.run(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        logger.debug("Command completed successfully: exit_code=%d", completed.returncode)
        if completed.stdout:
            logger.debug("Command stdout:\n%s", _truncate_for_log(completed.stdout))
        if completed.stderr:
            logger.debug("Command stderr:\n%s", _truncate_for_log(completed.stderr))
        return completed.stdout
    except FileNotFoundError as exc:
        logger.error("Command not found: %s", cmd_parts[0])
        raise ValueError(f"Command not found: {cmd_parts[0]}") from exc
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Command failed: exit_code=%d, command=%s",
            exc.returncode,
            cmd_str,
        )
        if exc.stdout:
            logger.debug("Command stdout (on error):\n%s", _truncate_for_log(exc.stdout))
        if exc.stderr:
            logger.debug("Command stderr (on error):\n%s", _truncate_for_log(exc.stderr))
        err = exc.stderr if exc.stderr is not None else exc.stdout
        message = err.strip() if err else str(exc)
        raise ValueError(
            f"Command failed with exit code {exc.returncode}: {message}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - re-raise as ValueError for API
        logger.exception("Failed to execute command: %s", cmd_str)
        raise ValueError(f"Failed to execute command: {exc}") from exc
