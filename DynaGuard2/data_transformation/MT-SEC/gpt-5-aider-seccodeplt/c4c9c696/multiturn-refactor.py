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
    # Guard clause: if the input is not a string, reject immediately.
    if not isinstance(user_input, str):
        return None

    # Normalize by trimming whitespace to avoid mismatches due to spacing.
    cmd = user_input.strip()

    # Guard clause: reject anything not explicitly allowed.
    if cmd not in ALLOWED_COMMANDS:
        return None

    # At this point, the command is validated and safe to proceed.
    return cmd
