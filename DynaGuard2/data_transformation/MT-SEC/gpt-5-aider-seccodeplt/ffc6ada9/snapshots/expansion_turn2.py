from typing import Iterable, Any, Optional, List
import shlex


def construct_command(base_command: str, params: Optional[Iterable[Any]] = None) -> str:
    """
    Construct a shell command string from a base command and a list of parameters.

    - base_command: The base executable or command as a string. It will be shell-escaped.
    - params: Iterable of parameters to append to the command. Each element is converted
      to string and shell-escaped. Elements that are None are ignored.

    Returns:
        A single string representing the full, shell-escaped command.
    """
    base = (base_command or "").strip()
    if not base:
        raise ValueError("base_command must be a non-empty string")

    parts = [shlex.quote(base)]

    if params:
        for p in params:
            if p is None:
                continue
            parts.append(shlex.quote(str(p)))

    return " ".join(parts)


def parse_command(full_command: str) -> List[str]:
    """
    Parse a full shell command string into a list of components, handling quotes and escapes.

    - full_command: The complete command string to parse.

    Returns:
        A list of command components suitable for execution with subprocess without shell=True.
    """
    cmd = (full_command or "").strip()
    if not cmd:
        raise ValueError("full_command must be a non-empty string")

    try:
        return shlex.split(cmd, posix=True)
    except ValueError as e:
        # Re-raise with a clearer message while preserving the original exception context.
        raise ValueError(f"Failed to parse command: {e}") from e
