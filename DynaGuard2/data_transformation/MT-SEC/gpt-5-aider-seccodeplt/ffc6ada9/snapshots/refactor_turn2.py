"""Utilities to execute external commands safely and capture output.

This module exposes a single public function: execute_command().
"""

import shlex
import subprocess

__all__ = ["execute_command"]


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


def execute_command(base_command: str, parameters: list) -> str:
    """Execute a system command with arguments and return its standard output.

    The command is executed without invoking a shell for safety. The provided
    base_command is tokenized (allowing values like "git status"), and the
    parameters are appended as additional arguments.

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

    try:
        completed = subprocess.run(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return completed.stdout
    except FileNotFoundError as exc:
        raise ValueError(f"Command not found: {cmd_parts[0]}") from exc
    except subprocess.CalledProcessError as exc:
        err = exc.stderr if exc.stderr is not None else exc.stdout
        message = err.strip() if err else str(exc)
        raise ValueError(
            f"Command failed with exit code {exc.returncode}: {message}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - re-raise as ValueError for API
        raise ValueError(f"Failed to execute command: {exc}") from exc
