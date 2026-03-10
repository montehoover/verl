import shlex
import subprocess
from typing import Sequence, Optional


def build_command_string(base_command: str, arguments: Optional[Sequence[str]] = None) -> str:
    """
    Build a shell-safe command string from a base command and its arguments.

    Args:
        base_command: The command/executable to invoke (e.g., 'ls' or '/usr/bin/grep').
        arguments: A sequence of argument strings. Non-string items will be converted to strings.

    Returns:
        A command string with tokens safely quoted for shell execution.

    Raises:
        ValueError: If base_command is empty.
    """
    if not base_command:
        raise ValueError("base_command must be a non-empty string")

    if arguments is None:
        arguments = []

    tokens = [str(base_command)] + [str(a) for a in arguments]

    if hasattr(shlex, "join"):
        return shlex.join(tokens)

    # Fallback for Python < 3.8
    return " ".join(shlex.quote(t) for t in tokens)


def execute_and_handle_errors(command: str) -> str:
    """
    Execute a command string and return its output, handling errors safely.

    The command string will be split into tokens with shlex.split and executed
    without invoking a shell to reduce injection risks.

    Args:
        command: The full command string to execute.

    Returns:
        The standard output from the command if it succeeds. If execution fails
        or the command exits with a non-zero status, an error message string
        describing the failure is returned instead.
    """
    if not command or not command.strip():
        return "Error: command string is empty."

    try:
        tokens = shlex.split(command)
    except ValueError as e:
        return f"Error: failed to parse command: {e}"

    if not tokens:
        return "Error: command string did not contain any tokens."

    try:
        result = subprocess.run(tokens, capture_output=True, text=True)
    except FileNotFoundError:
        return f"Error: command not found: {tokens[0]}"
    except PermissionError:
        return f"Error: permission denied: {tokens[0]}"
    except OSError as e:
        return f"Error: OS error while executing command: {e}"

    if result.returncode != 0:
        stderr_msg = (result.stderr or "").strip()
        if stderr_msg:
            return f"Error: command exited with status {result.returncode}: {stderr_msg}"
        return f"Error: command exited with status {result.returncode}."

    return result.stdout


def run_shell_command(cmd: str, arguments: Optional[Sequence[str]] = None) -> str:
    """
    Execute a command given a base command and a list of arguments, capturing stdout.

    Args:
        cmd: The base command/executable to run.
        arguments: A list or sequence of argument strings.

    Returns:
        The command's standard output as a string.

    Raises:
        ValueError: If the command is invalid (empty, not found, permission denied),
                    if execution fails (non-zero exit), or on other OS-related errors.
    """
    if not cmd or not str(cmd).strip():
        raise ValueError("cmd must be a non-empty string")

    if arguments is None:
        arguments = []

    # Build the argv list without invoking a shell to avoid injection risks.
    argv = [str(cmd)] + [str(a) for a in arguments]

    try:
        completed = subprocess.run(argv, capture_output=True, text=True, check=True)
        return completed.stdout
    except FileNotFoundError:
        raise ValueError(f"Command not found: {argv[0]}")
    except PermissionError:
        raise ValueError(f"Permission denied: {argv[0]}")
    except subprocess.CalledProcessError as e:
        stderr_msg = (e.stderr or "").strip()
        if stderr_msg:
            raise ValueError(f"Command failed with exit code {e.returncode}: {stderr_msg}")
        raise ValueError(f"Command failed with exit code {e.returncode}.")
    except OSError as e:
        raise ValueError(f"OS error while executing command: {e}")
