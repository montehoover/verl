# Setup: If ALLOWED_COMMANDS is not defined in the environment, fall back to TRUSTED_COMMANDS.
try:
    ALLOWED_COMMANDS  # type: ignore  # noqa: F401
except NameError:
    try:
        TRUSTED_COMMANDS  # type: ignore  # noqa: F401
    except NameError:
        TRUSTED_COMMANDS = ['ls', 'whoami', 'date']
    ALLOWED_COMMANDS = TRUSTED_COMMANDS  # type: ignore

def check_and_run_command(user_input):
    """
    Validates a user-provided command against the ALLOWED_COMMANDS list.

    Args:
        user_input (str): A command string provided by the user.

    Returns:
        str | None: The command string if allowed; otherwise, None.
    """
    if not isinstance(user_input, str):
        return None

    cmd = user_input.strip()
    return cmd if cmd in ALLOWED_COMMANDS else None
