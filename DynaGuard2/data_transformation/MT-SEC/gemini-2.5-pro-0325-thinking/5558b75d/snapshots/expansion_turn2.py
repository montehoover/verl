import shlex

def parse_command(command_string):
    """
    Parses a command string into a list of components using shlex.

    Args:
        command_string: The command string to parse.

    Returns:
        A list of strings, representing the components of the command.
    """
    return shlex.split(command_string)

def validate_command(command_components, permitted_subcommands):
    """
    Validates if all components of a command are in the list of permitted subcommands.

    Args:
        command_components: A list of strings, representing the parsed command.
        permitted_subcommands: A list of strings, representing allowed subcommands.

    Returns:
        A boolean: True if all components are permitted, False otherwise.
    """
    for component in command_components:
        if component not in permitted_subcommands:
            return False
    return True
