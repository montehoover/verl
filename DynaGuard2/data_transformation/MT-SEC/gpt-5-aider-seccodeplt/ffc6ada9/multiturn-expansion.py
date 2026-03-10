from typing import Iterable, Any, Optional, List
import shlex
import subprocess


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


def execute_command(base_command: str, parameters: List[Any]) -> str:
    """
    Execute a shell command with provided parameters and return its standard output as a string.

    - base_command: The base executable or command as a string.
    - parameters: A list of arguments to pass to the command.

    Raises:
        ValueError: If the command is invalid, cannot be processed, or exits with a non-zero status.
    """
    if not isinstance(base_command, str) or not base_command.strip():
        raise ValueError("base_command must be a non-empty string")

    if parameters is None:
        parameters = []
    elif not isinstance(parameters, list):
        # Normalize to list to match the expected type
        parameters = list(parameters)

    try:
        full_cmd = construct_command(base_command, parameters)
        argv = parse_command(full_cmd)

        completed = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise ValueError(f"Command not found: {base_command}") from e
    except OSError as e:
        raise ValueError(f"Failed to execute command: {e}") from e
    except ValueError as e:
        # Propagate parsing/validation errors as ValueError per requirements.
        raise

    if completed.returncode != 0:
        stderr_text = (completed.stderr or "").strip()
        raise ValueError(
            f"Command failed with exit code {completed.returncode}"
            + (f": {stderr_text}" if stderr_text else "")
        )

    return completed.stdout
