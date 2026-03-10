import shlex
from typing import Iterable, Any, List, Optional

def construct_command(main_command: str, args: Optional[Iterable[Any]] = None) -> str:
    """
    Build a shell-safe command string from a main command and a list/iterable of arguments.

    - main_command: The executable or shell command to run (e.g., "git", "/usr/bin/python").
    - args: Iterable of arguments. Each element will be converted to str and safely quoted.

    Returns a single string suitable for execution by a shell (e.g., via subprocess with shell=True).
    """
    if not isinstance(main_command, str) or not main_command:
        raise ValueError("main_command must be a non-empty string")

    if args is None:
        args_list: List[str] = []
    else:
        args_list = [str(a) for a in args]

    parts = [main_command] + args_list
    return " ".join(shlex.quote(p) for p in parts)
