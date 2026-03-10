import os
import shlex
import subprocess
from typing import List

__all__ = ["parse_command", "validate_subcommands", "validate_and_execute_command"]

def parse_command(command: str) -> List[str]:
    """
    Safely parse a shell command string into a list of arguments using shlex.

    This function:
    - Handles quotes and escaped spaces correctly.
    - Does not evaluate or execute the command.
    - Returns an empty list for empty/whitespace-only input.
    - Raises ValueError for unmatched quotes or invalid syntax.
    - Raises TypeError if the input is not a string.
    """
    if not isinstance(command, str):
        raise TypeError("command must be a string")

    if not command.strip():
        return []

    try:
        # POSIX mode provides standard shell-like parsing semantics.
        return shlex.split(command, posix=True)
    except ValueError as exc:
        # Typically raised for unmatched quotes.
        raise ValueError(f"Invalid command syntax: {exc}") from exc


def validate_subcommands(command: List[str], allowed_subcommands: List[str]) -> bool:
    """
    Validate that all elements in the parsed command are within the allowed subcommands.

    Args:
        command: A list of strings representing the parsed command (e.g., output of parse_command).
        allowed_subcommands: A list of permitted subcommand strings.

    Returns:
        True if every element in 'command' is present in 'allowed_subcommands', otherwise False.

    Raises:
        TypeError: If inputs are not lists of strings.
    """
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise TypeError("command must be a list of strings")
    if not isinstance(allowed_subcommands, list) or not all(isinstance(x, str) for x in allowed_subcommands):
        raise TypeError("allowed_subcommands must be a list of strings")

    allowed_set = set(allowed_subcommands)
    return all(token in allowed_set for token in command)


def validate_and_execute_command(shell_cmd: str, safe_subcommands: List[str]) -> str:
    """
    Parse, validate, and execute a shell command safely.

    Args:
        shell_cmd: The raw shell command string to parse and execute.
        safe_subcommands: A list of permissible subcommands/tokens.

    Returns:
        The command's stdout on success, or an error message string otherwise.

    Raises:
        TypeError: If inputs are not of the expected types.
    """
    if not isinstance(shell_cmd, str):
        raise TypeError("shell_cmd must be a string")
    if not isinstance(safe_subcommands, list) or not all(isinstance(x, str) for x in safe_subcommands):
        raise TypeError("safe_subcommands must be a list of strings")

    try:
        args = parse_command(shell_cmd)
    except ValueError as exc:
        return f"Invalid command syntax: {exc}"

    if not args:
        return "No command provided."

    if not validate_subcommands(args, safe_subcommands):
        return "Command contains disallowed subcommands."

    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            shell=False,
            env=os.environ.copy()
        )
    except FileNotFoundError:
        return f"Command not found: {args[0]}"
    except PermissionError:
        return f"Permission denied: {args[0]}"
    except OSError as exc:
        return f"Execution error: {exc}"

    if completed.returncode == 0:
        return completed.stdout
    else:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        message = stderr or stdout or f"Process exited with code {completed.returncode}"
        return f"Execution failed (exit {completed.returncode}): {message}"
