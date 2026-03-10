ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def is_command_allowed(command: str) -> str:
    """
    Checks if the given command is in the list of allowed commands and returns a message.

    Args:
        command: The command string to check.

    Returns:
        A string indicating if the command is allowed or not.
    """
    if command in ALLOWED_COMMANDS:
        return 'Command is allowed'
    else:
        return 'Command is not allowed'
