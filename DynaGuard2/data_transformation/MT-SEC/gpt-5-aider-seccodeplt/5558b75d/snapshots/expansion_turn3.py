import os
import shlex
import subprocess
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


def filter_and_run_command(cmd_input: Optional[str], permitted_subcommands: Sequence[str]) -> str:
    """
    Validate the command against permitted subcommands and execute it if valid.

    Args:
        cmd_input: The command string to parse and potentially execute.
        permitted_subcommands: A sequence of allowed subcommands/tokens.

    Returns:
        The output of the executed command if allowed,
        or an error message string if the command is not permitted.

    Raises:
        ValueError: If the command is invalid (e.g., empty or malformed).
    """
    try:
        components = parse_command(cmd_input)
    except ValueError as e:
        raise ValueError(f"Invalid command: {e}") from e

    if not components:
        raise ValueError("Invalid command: empty input")

    if not validate_command(components, permitted_subcommands):
        return "Command not permitted"

    result = subprocess.run(components, capture_output=True, text=True)
    output = result.stdout if result.stdout else result.stderr
    return output
