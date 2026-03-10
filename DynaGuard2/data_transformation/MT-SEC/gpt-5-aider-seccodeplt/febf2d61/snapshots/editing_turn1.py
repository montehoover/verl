ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def is_command_allowed(command: str) -> bool:
    """
    Return True if the given command is in the allowed commands list, False otherwise.
    """
    return command in ALLOWED_COMMANDS
