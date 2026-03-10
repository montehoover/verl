"""
This module provides a function to validate user-provided commands
against a predefined list of allowed commands.
"""

import logging

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
    # Configure logging
    logger = logging.getLogger(__name__)
    if not logger.handlers:  # Avoid adding multiple handlers if function is called multiple times
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    if cmd in ALLOWED_COMMANDS:
        logger.info(f"Command validation successful for: '{cmd}'")
        return cmd
    else:
        logger.warning(f"Command validation failed for: '{cmd}'. Command not allowed.")
        return None
