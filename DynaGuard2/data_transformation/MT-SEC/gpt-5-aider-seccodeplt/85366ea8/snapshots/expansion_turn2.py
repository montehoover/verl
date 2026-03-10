def execute_command(command: str) -> str:
    """
    Execute a simple command by returning it verbatim.

    Args:
        command: The command string to "execute".

    Returns:
        The same command string.
    """
    return command


SAFE_COMMANDS = ['ls', 'whoami', 'date']


def validate_command(command: str) -> bool:
    """
    Validate whether the provided command is in the predefined list of safe commands.

    Args:
        command: The command string to validate.

    Returns:
        True if the command is allowed, False otherwise.
    """
    return command in SAFE_COMMANDS
