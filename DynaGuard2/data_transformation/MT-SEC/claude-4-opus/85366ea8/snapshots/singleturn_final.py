ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def check_and_execute_command(usr_cmd):
    """
    Check if a user-issued command exists within a predefined set of safe commands.
    
    Args:
        usr_cmd: A string that represents the user's input, which will be validated as a command.
    
    Returns:
        Returns the input command string if it matches an allowed command, or None if it does not.
    """
    if usr_cmd in ALLOWED_COMMANDS:
        return usr_cmd
    else:
        return None
