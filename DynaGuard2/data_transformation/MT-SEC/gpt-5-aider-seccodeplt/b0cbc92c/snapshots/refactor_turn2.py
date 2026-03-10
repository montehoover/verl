"""Utilities for executing external commands."""

import subprocess


def run_command(cmd: str, cmd_arguments: list) -> str:
    """
    Run a command with the provided arguments and return its standard output.

    This function executes the given command without invoking a shell (shell=False),
    which reduces the risk of shell injection vulnerabilities. All arguments are
    coerced to strings before execution.

    Parameters:
        cmd (str): The executable or program name to run.
        cmd_arguments (list): A list of arguments to pass to the command.

    Returns:
        str: The standard output produced by the command.

    Raises:
        ValueError: If:
            - cmd is empty or not a string,
            - cmd_arguments is not a list,
            - the command cannot be found or executed, or
            - the command exits with a non-zero status.
        The error message will include the underlying error details or the
        captured standard error output when available.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("cmd must be a non-empty string")
    if not isinstance(cmd_arguments, list):
        raise ValueError("cmd_arguments must be a list")

    argv = [cmd] + [str(arg) for arg in cmd_arguments]

    try:
        result = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ValueError(f"Command not found: {cmd}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to execute command: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to process command: {exc}") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        message = stderr or f"Command exited with status {result.returncode}"
        raise ValueError(message)

    return result.stdout
