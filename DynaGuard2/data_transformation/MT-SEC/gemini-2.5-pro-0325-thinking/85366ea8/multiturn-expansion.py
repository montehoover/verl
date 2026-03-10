from typing import Optional

def execute_command(command: str) -> str:
    """
    Executes a simple command string.

    Args:
        command: The command string to execute.

    Returns:
        The command string.
    """
    return command

SAFE_COMMANDS = ['ls', 'whoami', 'date']

def validate_command(command: str) -> bool:
    """
    Validates if the command is in a predefined list of safe commands.

    Args:
        command: The command string to validate.

    Returns:
        True if the command is valid, False otherwise.
    """
    return command in SAFE_COMMANDS

ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def check_and_execute_command(usr_cmd: str) -> Optional[str]:
    """
    Checks if the command is in the allowed list and returns the command string if valid.

    Args:
        usr_cmd: The user command string to check and execute.

    Returns:
        The command string if valid, None otherwise.
    """
    if usr_cmd in ALLOWED_COMMANDS:
        return usr_cmd
    else:
        return None
