"""
Utilities for validating user-issued shell commands against a safelist.

The allowed commands are specified by the ALLOWED_COMMANDS constant.
This module exposes a single function, verify_and_run_command, that
checks whether a provided command is permitted. It does not execute
the command; it only verifies eligibility.
"""

import logging
from typing import Optional


# Safelist of commands the user is permitted to run.
ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def verify_and_run_command(command: str) -> Optional[str]:
    """
    Validate a user-issued command against the safelist and log the attempt.

    This function performs an exact string match against ALLOWED_COMMANDS.
    It does not execute the command; instead, it returns the command if
    it is permitted, or None if it is not. For observability, each check
    is logged with the outcome.

    Logging:
        - INFO: A message indicating whether the command was allowed or dismissed.

    Args:
        command: A string representing the user's input command.

    Returns:
        The original command string if it is allowed; otherwise, None.
    """
    # Initialize a logger within the function to record verification attempts.
    logger = logging.getLogger(__name__)

    # Reject non-string inputs early to keep the API predictable.
    if not isinstance(command, str):
        logger.info("Command dismissed: non-string input (%r).", command)
        return None

    # Return the command only if it is exactly in the safelist.
    if command in ALLOWED_COMMANDS:
        logger.info("Command allowed: %s", command)
        return command

    logger.info("Command dismissed: %s not in allowed commands.", command)
    return None
