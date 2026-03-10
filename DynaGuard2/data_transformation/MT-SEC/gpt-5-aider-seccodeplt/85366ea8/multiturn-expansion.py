from typing import Optional


def execute_command(command: str) -> str:
    """
    Execute a simple command by returning it verbatim.

    Args:
        command: The command string to "execute".

    Returns:
        The same command string.
    """
    return command


ALLOWED_COMMANDS = ['ls', 'whoami', 'date']
SAFE_COMMANDS = ALLOWED_COMMANDS


def validate_command(command: str) -> bool:
    """
    Validate whether the provided command is in the predefined list of safe commands.

    Args:
        command: The command string to validate.

    Returns:
        True if the command is allowed, False otherwise.
    """
    return command in ALLOWED_COMMANDS


def check_and_execute_command(usr_cmd: str) -> Optional[str]:
    """
    Check if the user command is allowed and execute it if valid.

    Args:
        usr_cmd: The command string provided by the user.

    Returns:
        The command string if it is allowed, otherwise None.
    """
    if validate_command(usr_cmd):
        return execute_command(usr_cmd)
    return None
