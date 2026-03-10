ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def verify_and_run_command(command: str):
    """
    Checks if a user-issued command exists within a predefined set of safe commands.

    Args:
        command: A string that represents the user's input.

    Returns:
        The input command string if it matches an allowed command, or None otherwise.
    """
    if command in ALLOWED_COMMANDS:
        return command
    else:
        return None
