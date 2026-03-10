import shlex
from typing import List

__all__ = ["parse_command", "validate_subcommands"]

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
