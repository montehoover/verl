import subprocess
import shlex

def run_basic_command(command: str) -> str:
    """
    Run a shell command with no arguments and return its standard output as a string.

    Parameters:
        command: The command to execute (executable name only, no arguments).

    Returns:
        The standard output produced by the command.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero status.
        FileNotFoundError: If the command is not found.
    """
    result = subprocess.run([command], capture_output=True, text=True, check=True)
    return result.stdout

def run_command_with_args(command: str, args: list[str]) -> str:
    """
    Run a shell command with arguments and return its standard output as a string.

    Parameters:
        command: The command to execute (may include subcommands).
        args: A list of arguments to pass to the command. Items will be converted to strings.

    Returns:
        The standard output produced by the command.

    Raises:
        ValueError: If the provided command is empty or only whitespace.
        subprocess.CalledProcessError: If the command exits with a non-zero status.
        FileNotFoundError: If the command is not found.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("Command must be a non-empty string.")

    command_parts = shlex.split(command)
    full_cmd = command_parts + [str(a) for a in args]

    result = subprocess.run(full_cmd, capture_output=True, text=True, check=True)
    return result.stdout

def call_system_command(sys_command: str, params: list[str]) -> str:
    """
    Execute a user-specified system command with arbitrary arguments and return its output.

    Parameters:
        sys_command: The primary command to execute (may include subcommands).
        params: A list of arguments to pass to the command.

    Returns:
        The standard output produced by the command as a string.

    Raises:
        ValueError: If the command is invalid or execution fails.
    """
    if not isinstance(sys_command, str) or not sys_command.strip():
        raise ValueError("Command must be a non-empty string.")

    try:
        command_parts = shlex.split(sys_command)
    except ValueError as e:
        raise ValueError(f"Invalid command string: {e}") from e

    full_cmd = command_parts + [str(p) for p in params]

    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise ValueError(f"Command not found: {command_parts[0]}") from e
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        msg = f"Command failed with exit code {e.returncode}"
        if stderr:
            msg += f": {stderr}"
        raise ValueError(msg) from e

    return result.stdout
