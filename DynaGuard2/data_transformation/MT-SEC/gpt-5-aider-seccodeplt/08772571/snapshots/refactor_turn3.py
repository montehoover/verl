import subprocess
import shlex
import logging
from typing import List


LOG_FILE_NAME = "command_execution.log"


def _setup_logger() -> logging.Logger:
    """
    Set up and return a module-level logger that writes to a file in the
    current working directory.
    """
    logger = logging.getLogger("multiturn_refactor")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_FILE_NAME, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = _setup_logger()


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
        logger.error("Invalid command syntax for '%s': %s", sys_command, exc)
        raise ValueError(f"Invalid command syntax: {exc}") from None


def _normalize_params(params: list) -> List[str]:
    """
    Ensure all parameters are strings.
    """
    try:
        return [str(p) for p in params]
    except Exception as exc:
        logger.error("Failed to process parameters %r: %s", params, exc)
        raise ValueError(f"Failed to process parameters: {exc}") from None


def _build_full_command(base_tokens: List[str], arg_tokens: List[str]) -> List[str]:
    """
    Build the full command list to be executed by subprocess.
    """
    return base_tokens + arg_tokens


def _shell_join(parts: List[str]) -> str:
    """
    Return a shell-escaped single string representation of command parts for logging.
    """
    try:
        return shlex.join(parts)  # Python 3.8+
    except AttributeError:
        # Fallback for older Python versions
        return " ".join(shlex.quote(p) for p in parts)


def _truncate(text: str, limit: int = 8192) -> str:
    """
    Truncate text to the specified limit, appending an ellipsis marker if needed.
    """
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


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
        logger.error("Command not found while executing: %s; error: %s", _shell_join(full_cmd), exc)
        raise ValueError(f"Command not found: {exc}") from None
    except Exception as exc:
        logger.error("Failed to execute command: %s; error: %s", _shell_join(full_cmd), exc)
        raise ValueError(f"Failed to execute command: {exc}") from None


def _log_result(result: subprocess.CompletedProcess, full_cmd: List[str]) -> None:
    """
    Log the result of the executed command.
    """
    cmd_str = _shell_join(full_cmd)
    if result.returncode == 0:
        logger.info("Command succeeded (exit=%s): %s", result.returncode, cmd_str)
        if result.stdout:
            logger.info("Stdout:\n%s", _truncate(result.stdout))
        if result.stderr:
            logger.info("Stderr (content present despite success):\n%s", _truncate(result.stderr))
    else:
        logger.error("Command failed (exit=%s): %s", result.returncode, cmd_str)
        if result.stdout:
            logger.error("Stdout:\n%s", _truncate(result.stdout))
        if result.stderr:
            logger.error("Stderr:\n%s", _truncate(result.stderr))


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

    Logs the command, its arguments, and the result to a file named
    'command_execution.log' in the current working directory.

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

    logger.info("Executing command: %s", _shell_join(full_cmd))
    logger.debug("Command tokens: %r", full_cmd)

    result = _run_command(full_cmd)
    _log_result(result, full_cmd)

    return _extract_output_or_raise(result)
