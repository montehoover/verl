ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def validate_and_execute_command(cmd):
    """Validate a user-provided command against allowed commands.
    
    Args:
        cmd (str): The command string to validate
        
    Returns:
        str or None: The command if valid, None otherwise
    """
    if cmd in ALLOWED_COMMANDS:
        return cmd
    return None
