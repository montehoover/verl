# List of commands that are allowed to be executed
ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def verify_and_run_command(command):
    """
    Verify if a user-issued command is within the predefined set of safe commands.
    
    This function checks whether the provided command string matches one of the
    commands in the ALLOWED_COMMANDS list. Only commands that are explicitly
    allowed will be validated for execution.
    
    Args:
        command (str): A string representing the user's input command to validate.
    
    Returns:
        str or None: The input command string if it matches an allowed command,
                     or None if the command is not in the allowed list.
    
    Examples:
        >>> verify_and_run_command('ls')
        'ls'
        >>> verify_and_run_command('rm')
        None
    """
    # Check if the command is in the allowed commands list
    if command in ALLOWED_COMMANDS:
        return command
    else:
        return None
