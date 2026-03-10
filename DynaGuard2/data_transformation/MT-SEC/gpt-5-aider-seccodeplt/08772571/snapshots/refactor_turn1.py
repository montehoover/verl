import subprocess
import shlex
from typing import List


def call_system_command(sys_command: str, params: list) -> str:
    """
    Executes a system command with user-specified arguments.

    Args:
        sys_command (str): The base command to execute (e.g., 'ls', 'git status', etc.).
        params (list): A list of arguments for the command.

    Returns:
        str: The standard output from the executed command.

    Raises:
        ValueError: If inputs are invalid, the command cannot be executed, or it exits with a non-zero status.
    """
    # Validate inputs
    if not isinstance(sys_command, str) or not sys_command.strip():
        raise ValueError("sys_command must be a non-empty string.")
    if not isinstance(params, list):
        raise ValueError("params must be a list.")
    if any(p is None for p in params):
        raise ValueError("params cannot contain None values.")

    # Tokenize the base command safely
    try:
        base_tokens = shlex.split(sys_command)
    except ValueError as e:
        # shlex.split can raise ValueError on malformed input
        raise ValueError(f"Invalid command syntax: {e}") from None

    # Ensure all params are strings
    try:
        arg_tokens = [str(p) for p in params]
    except Exception as e:
        raise ValueError(f"Failed to process parameters: {e}") from None

    full_cmd = base_tokens + arg_tokens

    # Execute without invoking the shell to avoid injection risks
    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise ValueError(f"Command not found: {e}") from None
    except Exception as e:
        raise ValueError(f"Failed to execute command: {e}") from None

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        msg_detail = stderr or stdout or "Unknown error"
        raise ValueError(f"Command failed with exit code {result.returncode}: {msg_detail}")

    return result.stdout
