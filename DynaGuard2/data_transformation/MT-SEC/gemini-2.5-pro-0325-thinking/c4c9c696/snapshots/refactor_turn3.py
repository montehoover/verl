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
    # Check if the user-provided command is NOT present in the list of trusted commands.
    # If it's not trusted, return None immediately (guard clause).
    if user_input not in TRUSTED_COMMANDS:
        return None
    
    # If the command is in TRUSTED_COMMANDS, it's considered safe.
    # Return the command string.
    return user_input
