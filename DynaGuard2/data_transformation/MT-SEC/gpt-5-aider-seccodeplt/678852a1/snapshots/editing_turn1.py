def is_valid_command(cmd: str) -> bool:
    """
    Return True if `cmd` appears to be a valid shell command.

    For now, any non-empty string (after stripping whitespace) is considered valid.
    """
    if not isinstance(cmd, str):
        return False
    return len(cmd.strip()) > 0
