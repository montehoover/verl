"""
This module provides a function to validate user-provided commands
against a predefined list of allowed commands.
"""

# A list of shell commands that are permitted to be executed.
ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def validate_and_execute_command(cmd: str) -> str | None:
    """
    Validates a user-provided command against a list of allowed commands.

    Args:
        cmd: The user-provided command to be validated.

    Returns:
        The command if it is in the list of allowed commands; otherwise None.
    """
    if cmd in ALLOWED_COMMANDS:
        return cmd
    else:
        return None
