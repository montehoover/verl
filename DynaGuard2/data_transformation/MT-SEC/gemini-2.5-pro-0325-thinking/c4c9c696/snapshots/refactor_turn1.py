TRUSTED_COMMANDS = ['ls', 'whoami', 'date']

def check_and_run_command(user_input: str):
    """
    Validates a command provided by the user to ensure it is included
    in a trusted list of commands. Only commands from this predefined
    list are allowed to proceed.

    Args:
        user_input: A string input representing a command from the user
                    to be validated.

    Returns:
        The command string if it is included in the TRUSTED_COMMANDS list;
        otherwise, it returns None.
    """
    if user_input in TRUSTED_COMMANDS:
        return user_input
    else:
        return None
