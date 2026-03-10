import subprocess
import shlex
from typing import List, Any


def run_shell_command(cmd: str, arguments: list) -> str:
    """
    Run a shell command with user-specified arguments.

    Args:
        cmd: The primary command to run (can include subcommands or flags).
        arguments: A list of arguments for the command.

    Returns:
        The standard output from the executed command as a string.

    Raises:
        ValueError: If the command is invalid or fails to execute.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("Invalid command: 'cmd' must be a non-empty string.")
    if not isinstance(arguments, list):
        raise ValueError("Invalid arguments: 'arguments' must be a list.")

    try:
        base_parts = shlex.split(cmd)
    except ValueError as e:
        raise ValueError(f"Invalid command: {e}") from e

    arg_parts = [str(a) for a in arguments]
    full_cmd = base_parts + arg_parts

    try:
        result = subprocess.run(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout
    except FileNotFoundError as e:
        raise ValueError(f"Command not found: {base_parts[0]}") from e
    except subprocess.CalledProcessError as e:
        # Command executed but returned a non-zero exit status
        err_output = e.stderr if e.stderr else e.stdout
        message = err_output.strip() if err_output else f"Command failed with exit code {e.returncode}"
        raise ValueError(message) from e
    except Exception as e:
        raise ValueError(str(e)) from e
