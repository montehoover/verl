"""
This module provides a function to validate user-issued commands
against a predefined list of safe commands.
"""

# A list of predefined safe commands that the user is allowed to execute.
ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def check_and_execute_command(usr_cmd: str):
    """Check if a user-issued command is in the list of allowed commands.

    Args:
        usr_cmd: A string representing the user's input command.

    Returns:
        The command string if it's in ALLOWED_COMMANDS, otherwise None.
    """
    if usr_cmd in ALLOWED_COMMANDS:
        return usr_cmd
    else:
        return None
