def log_command(command: str) -> str:
    """
    Logs a user-submitted command.

    Args:
        command: The command string submitted by the user.

    Returns:
        The command string.
    """
    # In a real application, this function would write to a log file,
    # database, or other persistent storage.
    # For now, it just returns the command.
    return command

def is_command_trusted(command: str) -> bool:
    """
    Checks if a command is in the list of trusted commands.

    Args:
        command: The command string to check.

    Returns:
        True if the command is trusted, False otherwise.
    """
    trusted_commands = ['ls', 'whoami', 'date']
    return command in trusted_commands
