"""
Utilities for validating user-provided commands against a predefined
allowlist. The main entry point is `check_and_run_command`, which ensures a
command is permitted before allowing it to proceed in the application flow.

This module expects an `ALLOWED_COMMANDS` collection to be available at
runtime. If it is not defined by the embedding application, it will fall back
to `TRUSTED_COMMANDS`, which contains a minimal, safe default set.
"""

from typing import Optional

# -----------------------------------------------------------------------------
# Configuration/bootstrap:
# - Prefer an externally provided ALLOWED_COMMANDS.
# - If unavailable, fall back to TRUSTED_COMMANDS.
# - If TRUSTED_COMMANDS is also unavailable, define a safe default.
# -----------------------------------------------------------------------------
try:
    ALLOWED_COMMANDS  # type: ignore[name-defined]
except NameError:
    try:
        TRUSTED_COMMANDS  # type: ignore[name-defined]
    except NameError:
        # Safe defaults if no trusted/allowed commands are provided upstream.
        TRUSTED_COMMANDS = ['ls', 'whoami', 'date']

    # Use TRUSTED_COMMANDS as the allowed set when ALLOWED_COMMANDS is absent.
    ALLOWED_COMMANDS = TRUSTED_COMMANDS  # type: ignore[assignment]


def check_and_run_command(user_input: str) -> Optional[str]:
    """
    Validate a user-provided command against the allowlist.

    This function ensures that only commands explicitly included in
    `ALLOWED_COMMANDS` are permitted to proceed. It performs basic normalization
    (trimming surrounding whitespace) before validation.

    Args:
        user_input (str): The raw command string provided by the user.

    Returns:
        Optional[str]: The normalized command string if it is present in
        `ALLOWED_COMMANDS`; otherwise, None.

    Examples:
        >>> check_and_run_command(" ls ")
        'ls'
        >>> check_and_run_command("rm -rf /")
        None
    """
    # Defensive type check: callers should pass a string, but this guards
    # against accidental misuse or unexpected input types.
    if not isinstance(user_input, str):
        return None

    # Normalize the user input by removing leading/trailing whitespace to
    # avoid false negatives caused by incidental spacing.
    cmd = user_input.strip()

    # Only allow the command if it exactly matches one of the allowed entries.
    # If not present, return None to indicate rejection.
    return cmd if cmd in ALLOWED_COMMANDS else None
