import shlex
from typing import List

__all__ = ["parse_command"]

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
