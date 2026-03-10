import subprocess


def execute_command(command: str) -> str:
    """
    Execute a shell command and return its output as a string.

    Parameters:
        command (str): The command to execute.

    Returns:
        str: The command's standard output if successful; otherwise, the error output.
    """
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout
    return result.stderr if result.stderr else result.stdout
