ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def verify_and_run_command(command):
    """
    Validate a user-issued command against a predefined set of allowed commands.

    Args:
        command (str): The user's input command to validate.

    Returns:
        str | None: The input command string if it matches an allowed command,
        or None if it does not.
    """
    if not isinstance(command, str):
        return None

    candidate = command.strip()
    return candidate if candidate in ALLOWED_COMMANDS else None
