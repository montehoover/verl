import subprocess
import shlex


def _validate_inputs(cmd: str, arguments: list) -> None:
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("Invalid command: 'cmd' must be a non-empty string.")
    if not isinstance(arguments, list):
        raise ValueError("Invalid arguments: 'arguments' must be a list.")


def _split_base_command(cmd: str) -> list:
    try:
        return shlex.split(cmd)
    except ValueError as e:
        raise ValueError(f"Invalid command: {e}") from e


def _assemble_command(cmd: str, arguments: list) -> list:
    base_parts = _split_base_command(cmd)
    arg_parts = [str(a) for a in arguments]
    return base_parts + arg_parts


def _execute_command(full_cmd: list) -> str:
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
        raise ValueError(f"Command not found: {full_cmd[0]}") from e
    except subprocess.CalledProcessError as e:
        err_output = e.stderr if e.stderr else e.stdout
        message = err_output.strip() if err_output else f"Command failed with exit code {e.returncode}"
        raise ValueError(message) from e
    except Exception as e:
        raise ValueError(str(e)) from e


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
    _validate_inputs(cmd, arguments)
    full_cmd = _assemble_command(cmd, arguments)
    return _execute_command(full_cmd)
