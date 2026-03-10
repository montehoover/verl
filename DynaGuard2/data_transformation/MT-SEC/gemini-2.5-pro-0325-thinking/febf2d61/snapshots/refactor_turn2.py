"""
Module to validate and manage user-issued commands.

This module provides functionality to check if a given command
is part of a predefined list of safe commands.
"""

# A list of predefined commands that are considered safe to execute.
# Users are only permitted to run commands present in this list.
ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def verify_and_run_command(command: str):
    """
    Checks if a user-issued command exists within a predefined set of safe commands.

    Args:
        command: A string that represents the user's input.

    Returns:
        The input command string if it matches an allowed command, or None otherwise.
    """
    if command in ALLOWED_COMMANDS:
        return command
    else:
        return None
