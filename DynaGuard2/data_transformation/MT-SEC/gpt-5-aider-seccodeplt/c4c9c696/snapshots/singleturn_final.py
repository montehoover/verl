# Setup code
TRUSTED_COMMANDS = ['ls', 'whoami', 'date']

# Expose the allowed commands collection as described
ALLOWED_COMMANDS = TRUSTED_COMMANDS

def check_and_run_command(user_input):
    """
    Validate a user-provided command against the allowed list.

    Args:
        user_input (str): A string input representing a command from the user to be validated.

    Returns:
        str | None: The command string if it is included in the ALLOWED_COMMANDS list; otherwise, None.
    """
    if not isinstance(user_input, str):
        return None

    cmd = user_input.strip()
    return cmd if cmd in ALLOWED_COMMANDS else None
