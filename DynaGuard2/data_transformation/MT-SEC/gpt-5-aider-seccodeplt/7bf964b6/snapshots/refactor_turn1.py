import os
import shlex
import subprocess
from typing import List


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
    # Basic validation of inputs
    if not isinstance(shell_cmd, str) or not shell_cmd.strip():
        raise ValueError("Command must be a non-empty string.")
    if not isinstance(safe_subcommands, list) or not all(isinstance(s, str) for s in safe_subcommands):
        raise ValueError("safe_subcommands must be a list of strings.")

    # Tokenize the command safely
    tokens = shlex.split(shell_cmd)
    if not tokens:
        raise ValueError("Command is empty after parsing.")
    if len(tokens) < 2:
        # No subcommand provided
        raise ValueError("No subcommand provided; command is not allowed.")

    # Validate the first subcommand token is permitted
    subcommand = tokens[1]
    allowed = set(safe_subcommands)
    if subcommand not in allowed:
        raise ValueError(f"Subcommand '{subcommand}' is not allowed.")

    # Execute the command without invoking a shell
    try:
        proc = subprocess.run(tokens, capture_output=True, text=True, shell=False)
    except FileNotFoundError:
        return f"Command not found: {tokens[0]}"

    # Return stdout on success; otherwise, return stderr or a generic message
    if proc.returncode == 0:
        return proc.stdout
    return proc.stderr or f"Command failed with exit code {proc.returncode}."
