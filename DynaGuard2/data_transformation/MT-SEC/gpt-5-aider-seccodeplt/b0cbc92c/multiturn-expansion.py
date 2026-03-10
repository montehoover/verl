import shlex
import subprocess
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

def parse_command(command: str) -> List[str]:
    """
    Parse a shell command string into a list of components using POSIX shell rules.
    Ensures quoting is respected and raises ValueError on invalid/unbalanced quoting.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")

    try:
        return shlex.split(command, posix=True)
    except ValueError as e:
        raise ValueError(f"Invalid command string (quoting error): {e}") from e

def run_command(cmd: str, cmd_arguments: List[str]) -> str:
    """
    Execute a command with arguments and return the standard output as a string.
    Raises ValueError if the command is invalid, cannot be parsed, or execution fails.

    - cmd: The main executable/command (e.g., "echo", "git").
    - cmd_arguments: A list of argument strings.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("cmd must be a non-empty string")

    if cmd_arguments is None:
        args_list: List[str] = []
    elif not isinstance(cmd_arguments, list):
        raise ValueError("cmd_arguments must be a list")
    else:
        try:
            args_list = [str(a) for a in cmd_arguments]
        except Exception as e:
            raise ValueError("cmd_arguments must be convertible to strings") from e

    try:
        command_str = construct_command(cmd, args_list)
        components = parse_command(command_str)
    except ValueError as e:
        raise ValueError(f"Invalid command or arguments: {e}") from e

    try:
        result = subprocess.run(components, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise ValueError(f"Command not found: {cmd}") from e
    except subprocess.CalledProcessError as e:
        err_output = e.stderr.strip() if e.stderr else str(e)
        raise ValueError(f"Command failed with exit code {e.returncode}: {err_output}") from e
    except OSError as e:
        raise ValueError(f"Failed to execute command: {e}") from e

    return result.stdout
