ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def check_and_execute_command(usr_cmd):
    """
    Validates the user-issued command against a predefined set of safe commands.

    Args:
        usr_cmd (str): A string representing the user's input command.

    Returns:
        str | None: The input command string if it matches an allowed command, or None if it does not.
    """
    if not isinstance(usr_cmd, str):
        return None

    cmd = usr_cmd.strip()
    return cmd if cmd in ALLOWED_COMMANDS else None
