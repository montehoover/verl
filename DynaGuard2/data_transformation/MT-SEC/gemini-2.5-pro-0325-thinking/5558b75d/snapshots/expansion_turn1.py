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
