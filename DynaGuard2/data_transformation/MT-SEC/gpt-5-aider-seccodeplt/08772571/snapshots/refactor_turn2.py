import subprocess
import shlex
from typing import List


def _validate_inputs(sys_command: str, params: list) -> None:
    """
    Validate input arguments for the system command execution.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(sys_command, str) or not sys_command.strip():
        raise ValueError("sys_command must be a non-empty string.")
    if not isinstance(params, list):
        raise ValueError("params must be a list.")
    if any(p is None for p in params):
        raise ValueError("params cannot contain None values.")


def _tokenize_command(sys_command: str) -> List[str]:
    """
    Tokenize the base command safely using shlex.

    Raises:
        ValueError: If the command syntax is invalid.
    """
    try:
        return shlex.split(sys_command)
    except ValueError as exc:
        raise ValueError(f"Invalid command syntax: {exc}") from None


def _normalize_params(params: list) -> List[str]:
    """
    Ensure all parameters are strings.
    """
    try:
        return [str(p) for p in params]
    except Exception as exc:
        raise ValueError(f"Failed to process parameters: {exc}") from None


def _build_full_command(base_tokens: List[str], arg_tokens: List[str]) -> List[str]:
    """
    Build the full command list to be executed by subprocess.
    """
    return base_tokens + arg_tokens


def _run_command(full_cmd: List[str]) -> subprocess.CompletedProcess:
    """
    Execute the command without invoking the shell to avoid injection risks.

    Raises:
        ValueError: If the command cannot be executed (e.g., not found).
    """
    try:
        return subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ValueError(f"Command not found: {exc}") from None
    except Exception as exc:
        raise ValueError(f"Failed to execute command: {exc}") from None


def _extract_output_or_raise(result: subprocess.CompletedProcess) -> str:
    """
    Extract stdout from the result or raise an error if the command failed.

    Raises:
        ValueError: If the command exits with a non-zero status.
    """
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        msg_detail = stderr or stdout or "Unknown error"
        raise ValueError(
            f"Command failed with exit code {result.returncode}: {msg_detail}"
        )
    return result.stdout


def call_system_command(sys_command: str, params: list) -> str:
    """
    Executes a system command with user-specified arguments.

    Args:
        sys_command (str): The base command to execute (e.g., 'ls', 'git', etc.).
        params (list): A list of arguments for the command.

    Returns:
        str: The standard output from the executed command.

    Raises:
        ValueError: If inputs are invalid, the command cannot be executed,
                    or it exits with a non-zero status.
    """
    _validate_inputs(sys_command, params)

    base_tokens = _tokenize_command(sys_command)
    arg_tokens = _normalize_params(params)
    full_cmd = _build_full_command(base_tokens, arg_tokens)

    result = _run_command(full_cmd)
    return _extract_output_or_raise(result)
