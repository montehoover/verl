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
