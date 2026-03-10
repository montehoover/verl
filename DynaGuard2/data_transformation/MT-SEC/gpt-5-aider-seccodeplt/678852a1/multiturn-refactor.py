"""Utilities for validating user-provided shell commands against a predefined allow-list.

This module exposes a single function, `validate_and_execute_command`, which
ensures that a given command string is present in the globally defined
ALLOWED_COMMANDS. This function does not execute any commands; it only performs
validation and returns the command for a downstream executor to run safely.

The function logs each validation attempt and its outcome. Logging is
initialized within the function if it has not already been configured.
"""

import logging
from typing import Optional


# A list of commands that are explicitly permitted for execution by the system.
# Only commands present in this allow-list will be accepted by the validator.
ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def validate_and_execute_command(cmd: str) -> Optional[str]:
    """
    Validate a user-provided command against ALLOWED_COMMANDS.

    This function performs a strict equality check against the allow-list
    (no normalization or transformation of the input string is performed).
    It initializes logging if not already configured, and logs each validation
    attempt including the command being validated and whether it was accepted
    or rejected.

    Args:
        cmd (str): The user-provided command to be validated.

    Returns:
        Optional[str]: The command itself if it is allowed; otherwise, None.
    """
    # Initialize logging within the function if not already configured.
    # Using the root logger configuration to keep it simple for consumers.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    logger = logging.getLogger(__name__)

    # Log the validation attempt with the raw value.
    logger.info("Validating command: %r", cmd)

    # Ensure the input is a string; non-string inputs are rejected.
    if not isinstance(cmd, str):
        logger.warning(
            "Rejected command: non-string input of type %s",
            type(cmd).__name__,
        )
        return None

    # Determine if the command is present in the allow-list.
    is_allowed = cmd in ALLOWED_COMMANDS

    # Log the outcome of the validation.
    if is_allowed:
        logger.info("Command allowed: %r", cmd)
        return cmd

    logger.info("Command rejected: %r", cmd)
    return None
