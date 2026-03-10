import os
import shlex
from typing import List


def validate_and_execute_command(shell_cmd: str, safe_subcommands: List[str]) -> str:
    """
    Execute a shell command only if all of its tokens are included in the provided
    list of safe subcommands.

    Args:
        shell_cmd: The full shell command string to execute (e.g., "ls -l").
        safe_subcommands: A list of allowed tokens (command and subcommands/flags/args).
                          Every token parsed from shell_cmd must appear in this list.

    Returns:
        The stdout string from the executed command.

    Raises:
        ValueError: If shell_cmd is not a string, is empty/only whitespace, if
                    safe_subcommands is not a list of strings, or if any token
                    in shell_cmd is not present in safe_subcommands.
    """
    # Basic type and value checks
    if not isinstance(shell_cmd, str):
        raise ValueError("shell_cmd must be a string.")

    if not isinstance(safe_subcommands, list):
        raise ValueError("safe_subcommands must be a list of strings.")

    # Normalize and validate safe_subcommands
    try:
        allowed_tokens = {str(token) for token in safe_subcommands}
    except Exception as exc:
        raise ValueError("safe_subcommands must contain values convertible to strings.") from exc

    # Parse the shell command into tokens safely
    tokens = shlex.split(shell_cmd, posix=True)

    if not tokens:
        raise ValueError("No command provided to execute.")

    # Ensure every token is allowed
    disallowed = [tok for tok in tokens if tok not in allowed_tokens]
    if disallowed:
        raise ValueError(f"Disallowed subcommand(s) or argument(s): {', '.join(disallowed)}")

    # Reconstruct a safe command string and execute
    # Quote each token to preserve exact tokenization and avoid shell interpretation issues.
    safe_cmd_str = " ".join(shlex.quote(tok) for tok in tokens)

    # Execute the command and return its output
    # Note: os.popen captures stdout only.
    return os.popen(safe_cmd_str).read()
