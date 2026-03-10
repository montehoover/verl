def is_valid_command(cmd: str) -> bool:
    """
    Checks if a given command string is a valid shell command.
    For now, any non-empty string is considered valid.
    """
    return bool(cmd)
