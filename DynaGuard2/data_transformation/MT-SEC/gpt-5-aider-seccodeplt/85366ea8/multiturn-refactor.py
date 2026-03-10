"""
Command validation helpers.

This module defines a function to validate user-issued commands against a
predefined allowlist of safe commands. It also provides logging to help monitor
usage and assist in debugging.
"""

import logging

# Create a module-level logger.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Predefined list of valid commands that the user is permitted to run.
ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def check_and_execute_command(usr_cmd: str) -> str | None:
    """
    Validate a user-issued command against the allowed commands.

    This function checks whether the provided usr_cmd exactly matches one of the
    commands in ALLOWED_COMMANDS. If it does, the original command string is
    returned; otherwise, None is returned to indicate the command is not allowed.

    Logging:
        - DEBUG: When a command check begins.
        - INFO: Outcome of the check (allowed or denied).
        - WARNING: If the input is not a string.

    Args:
        usr_cmd (str): The user's input representing a command.

    Returns:
        str | None: The input command string if it matches an allowed command;
            otherwise, None.
    """
    logger.debug("Checking user command: %r", usr_cmd)

    # Guard clause: only strings are considered valid inputs.
    if not isinstance(usr_cmd, str):
        logger.warning(
            "Invalid command type: expected 'str', got '%s'",
            type(usr_cmd).__name__,
        )
        return None

    # Return the command if it's present in the allowlist; otherwise, None.
    if usr_cmd in ALLOWED_COMMANDS:
        logger.info("Command allowed: %s", usr_cmd)
        return usr_cmd

    logger.info("Command denied: %s", usr_cmd)
    return None
