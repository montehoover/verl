ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def check_and_execute_command(usr_cmd: str) -> str | None:
    """
    Check whether the provided command is in the list of allowed commands.

    Args:
        usr_cmd: A string that represents the user's input command.

    Returns:
        The input command string if it matches an allowed command; otherwise, None.
    """
    if not isinstance(usr_cmd, str):
        return None

    return usr_cmd if usr_cmd in ALLOWED_COMMANDS else None
