ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def verify_and_run_command(command):
    """
    Check if a user-issued command exists within a predefined set of safe commands.
    
    Args:
        command: A string that represents the user's input, which will be validated as a command.
    
    Returns:
        Returns the input command string if it matches an allowed command, or None if it does not.
    """
    if command in ALLOWED_COMMANDS:
        return command
    return None
