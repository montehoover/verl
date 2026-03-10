from typing import Optional

ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def check_and_execute_command(usr_cmd: str) -> Optional[str]:
    """
    Checks if a user-issued command exists within a predefined set of safe commands.

    Args:
        usr_cmd: The command string issued by the user.

    Returns:
        The command string if it is valid and allowed, None otherwise.
    """
    if usr_cmd in ALLOWED_COMMANDS:
        return usr_cmd
    return None
