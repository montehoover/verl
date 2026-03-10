import subprocess
from typing import Optional


ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def is_command_allowed(command: str) -> bool:
    """
    Check if the provided command is allowed.

    Parameters:
        command (str): The command to check.

    Returns:
        bool: True if the command is in the allowed list; otherwise, False.
    """
    return command.strip() in ALLOWED_COMMANDS


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


def verify_and_run_command(command: str) -> Optional[str]:
    """
    Verify whether the command is allowed and return the command string if valid.
    Returns None if the command is not allowed.
    """
    if is_command_allowed(command):
        return command
    return None
