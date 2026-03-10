ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def validate_and_execute_command(cmd):
    """
    Validates a user-provided command against ALLOWED_COMMANDS.

    Args:
        cmd (str): The user-provided command to be validated.

    Returns:
        str | None: The command if allowed; otherwise None.
    """
    if not isinstance(cmd, str):
        return None

    return cmd if cmd in ALLOWED_COMMANDS else None
