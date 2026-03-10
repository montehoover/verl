from typing import Iterable, Any, Optional
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
