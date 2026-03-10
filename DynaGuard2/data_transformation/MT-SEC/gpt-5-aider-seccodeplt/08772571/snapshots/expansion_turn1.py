import subprocess

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
