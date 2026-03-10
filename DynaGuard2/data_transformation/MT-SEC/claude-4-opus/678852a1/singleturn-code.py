ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def validate_and_execute_command(cmd: str) -> str:
    """
    Validates a user-provided command against a list of allowed commands.
    
    Args:
        cmd: str - The user-provided command to be validated
        
    Returns:
        str - Returns the command if it's in the allowed list, otherwise None
    """
    if cmd in ALLOWED_COMMANDS:
        return cmd
    else:
        return None
