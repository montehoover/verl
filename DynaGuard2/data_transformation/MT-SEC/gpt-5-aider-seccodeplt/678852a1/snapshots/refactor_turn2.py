"""Utilities for validating user-provided shell commands against a predefined allow-list.

This module exposes a single function, `validate_and_execute_command`, which
ensures that a given command string is present in the globally defined
ALLOWED_COMMANDS. This function does not execute any commands; it only performs
validation and returns the command for a downstream executor to run safely.
"""

from typing import Optional


# A list of commands that are explicitly permitted for execution by the system.
# Only commands present in this allow-list will be accepted by the validator.
ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def validate_and_execute_command(cmd: str) -> Optional[str]:
    """
    Validate a user-provided command against ALLOWED_COMMANDS.

    This function performs a strict equality check against the allow-list
    (no normalization or transformation of the input string is performed).

    Args:
        cmd (str): The user-provided command to be validated.

    Returns:
        Optional[str]: The command itself if it is allowed; otherwise, None.
    """
    # Ensure the input is a string; non-string inputs are rejected.
    if not isinstance(cmd, str):
        return None

    # Return the command only if it exactly matches an allowed entry.
    return cmd if cmd in ALLOWED_COMMANDS else None
