# Define a list of trusted commands that are considered safe for execution.
# This list acts as an allowlist for user-provided commands.
TRUSTED_COMMANDS = ['ls', 'whoami', 'date']

def check_and_run_command(user_input: str):
    """
    Validates a command provided by the user against a predefined list of
    trusted commands.

    This function ensures that only commands explicitly listed in
    TRUSTED_COMMANDS are considered valid. This is a security measure
    to prevent the execution of arbitrary or malicious commands.

    Args:
        user_input (str): The command string input by the user.
                          This input will be checked against the
                          TRUSTED_COMMANDS list.

    Returns:
        str | None: The command string if it is found in the
                    TRUSTED_COMMANDS list, indicating it is a valid
                    and allowed command. Returns None if the command
                    is not in the list, indicating it is not allowed.
    """
    # Check if the user-provided command is present in the list of trusted commands.
    if user_input in TRUSTED_COMMANDS:
        # If the command is trusted, return the command string.
        return user_input
    else:
        # If the command is not trusted, return None.
        return None
