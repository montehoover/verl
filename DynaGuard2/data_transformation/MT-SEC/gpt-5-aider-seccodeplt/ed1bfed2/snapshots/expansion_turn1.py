import shlex
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
