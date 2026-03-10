ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def validate_and_execute_command(cmd: str) -> str | None:
    """
    Validates a user-provided command against a list of allowed commands.
    Returns the command if it's valid, otherwise None.
    """
    if cmd in ALLOWED_COMMANDS:
        return cmd
    return None
