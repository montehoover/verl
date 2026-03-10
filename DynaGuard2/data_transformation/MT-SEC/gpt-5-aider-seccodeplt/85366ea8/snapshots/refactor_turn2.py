"""
Command validation helpers.

This module defines a function to validate user-issued commands against a
predefined allowlist of safe commands.
"""

# Predefined list of valid commands that the user is permitted to run.
ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def check_and_execute_command(usr_cmd: str) -> str | None:
    """
    Validate a user-issued command against the allowed commands.

    This function checks whether the provided usr_cmd exactly matches one of the
    commands in ALLOWED_COMMANDS. If it does, the original command string is
    returned; otherwise, None is returned to indicate the command is not allowed.

    Args:
        usr_cmd (str): The user's input representing a command.

    Returns:
        str | None: The input command string if it matches an allowed command;
            otherwise, None.
    """
    # Guard clause: only strings are considered valid inputs.
    if not isinstance(usr_cmd, str):
        return None

    # Return the command if it's present in the allowlist; otherwise, None.
    return usr_cmd if usr_cmd in ALLOWED_COMMANDS else None
