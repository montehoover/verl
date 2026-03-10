import shlex
from typing import List, Optional


def parse_command(command: Optional[str]) -> List[str]:
    """
    Safely parse a command string into its components using shell-like syntax.

    - Respects quotes, escapes, and whitespace similar to POSIX shells.
    - Returns an empty list for None or empty/whitespace-only input.
    - Raises ValueError on malformed input (e.g., unmatched quotes).

    Args:
        command: The command string to parse.

    Returns:
        A list of parsed components.
    """
    if command is None:
        return []

    text = command.strip()
    if not text:
        return []

    # posix=True enables standard shell-like parsing rules.
    return shlex.split(text, posix=True)
