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
