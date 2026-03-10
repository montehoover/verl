def validate_command(command_string: str, permitted_subcommands: list[str]) -> bool:
    """
    Checks if a given command string contains only allowed subcommands.

    Args:
        command_string: The command string to validate.
        permitted_subcommands: A list of permitted subcommands.

    Returns:
        True if the command is valid, otherwise False.
    """
    subcommands = command_string.split()
    for subcommand in subcommands:
        if subcommand not in permitted_subcommands:
            return False
    return True
