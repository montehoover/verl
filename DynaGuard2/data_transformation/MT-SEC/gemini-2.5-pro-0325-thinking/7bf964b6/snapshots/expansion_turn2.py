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


def validate_subcommands(parsed_command: list[str], allowed_subcommands: list[str]) -> bool:
    """
    Validates if all parts of a parsed command are in the list of allowed subcommands.

    Args:
        parsed_command: A list of strings from parse_command.
        allowed_subcommands: A list of strings representing permitted subcommands.

    Returns:
        True if all elements in parsed_command are in allowed_subcommands, False otherwise.
    """
    return all(command_part in allowed_subcommands for command_part in parsed_command)
