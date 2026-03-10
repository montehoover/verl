def is_command_valid(command: str) -> bool:
    """
    Checks if the given command is in a predefined list of commands.

    Args:
        command: The command string to validate.

    Returns:
        True if the command is valid, False otherwise.
    """
    valid_commands = ["start", "stop", "status", "restart"]  # Example list of commands
    return command in valid_commands
