import shlex

def parse_command(command_string: str) -> list[str]:
    """
    Parses a shell command string into a list of arguments using shlex.

    Args:
        command_string: The shell command to parse.

    Returns:
        A list of strings, representing the parsed command arguments.
    """
    return shlex.split(command_string)
