import shlex
from typing import Iterable, List, Optional, Sequence


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


def validate_command(components: Sequence[str], permitted_subcommands: Iterable[str]) -> bool:
    """
    Validate that all components are within the permitted subcommands list.

    Args:
        components: The parsed command components to validate.
        permitted_subcommands: Iterable of allowed subcommands/tokens.

    Returns:
        True if every component is permitted, otherwise False.
    """
    permitted_set = set(permitted_subcommands)
    return all(component in permitted_set for component in components)
