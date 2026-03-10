import subprocess

def run_shell_command(base_command: str, args: list[str]) -> str:
    """
    Executes a shell command and returns its standard output.

    Args:
        base_command: The base command string (e.g., "ls").
        args: A list of arguments (e.g., ["-l", "/home"]).

    Returns:
        The standard output of the command as a string.
    """
    command = [base_command] + args
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout
