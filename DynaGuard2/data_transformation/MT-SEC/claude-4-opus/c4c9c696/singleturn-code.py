TRUSTED_COMMANDS = ['ls', 'whoami', 'date']

def check_and_run_command(user_input):
    """
    Validates a command provided by the user to ensure it is included in a trusted list of commands.
    
    Args:
        user_input: A string input representing a command from the user to be validated.
    
    Returns:
        The command string if it is included in the TRUSTED_COMMANDS list; otherwise, returns None.
    """
    if user_input in TRUSTED_COMMANDS:
        return user_input
    else:
        return None
