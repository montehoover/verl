ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def is_command_allowed(command: str) -> str:
    """
    Return a message indicating whether the given command is in the allowed commands list.
    """
    if command in ALLOWED_COMMANDS:
        return 'Command is allowed'
    return 'Command is not allowed'
