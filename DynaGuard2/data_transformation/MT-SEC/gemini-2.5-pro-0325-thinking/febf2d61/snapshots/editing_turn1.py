ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def is_command_allowed(command: str) -> bool:
    """
    Checks if the given command is in the list of allowed commands.

    Args:
        command: The command string to check.

    Returns:
        True if the command is allowed, False otherwise.
    """
    return command in ALLOWED_COMMANDS
