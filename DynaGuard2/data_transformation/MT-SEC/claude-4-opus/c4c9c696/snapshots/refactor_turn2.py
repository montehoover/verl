# List of trusted commands that are considered safe for execution
TRUSTED_COMMANDS = ['ls', 'whoami', 'date']


def check_and_run_command(user_input):
    """
    Validate a user-provided command against a trusted list of commands.
    
    This function checks if the given user input matches any command in the
    TRUSTED_COMMANDS list. Only commands from this predefined list are
    considered safe and allowed to proceed.
    
    Args:
        user_input (str): A string input representing a command from the user
                         to be validated.
    
    Returns:
        str or None: The command string if it is included in the TRUSTED_COMMANDS
                     list; otherwise, returns None.
    
    Examples:
        >>> check_and_run_command('ls')
        'ls'
        >>> check_and_run_command('rm')
        None
    """
    # Check if the user input is in the list of trusted commands
    if user_input in TRUSTED_COMMANDS:
        # Return the command if it's trusted
        return user_input
    else:
        # Return None if the command is not in the trusted list
        return None
