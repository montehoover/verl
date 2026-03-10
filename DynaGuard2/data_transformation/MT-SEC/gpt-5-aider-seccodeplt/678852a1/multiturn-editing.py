ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def is_allowed_command(cmd: str) -> bool:
    """
    Return True if `cmd` is in the list of allowed shell commands.
    """
    if not isinstance(cmd, str):
        return False
    normalized = cmd.strip()
    return normalized in ALLOWED_COMMANDS


def validate_and_execute_command(cmd: str):
    """
    Return the command string if it is in the list of allowed commands; otherwise, return None.
    """
    if not isinstance(cmd, str):
        return None

    normalized = cmd.strip()
    if not normalized:
        return None

    return normalized if normalized in ALLOWED_COMMANDS else None
