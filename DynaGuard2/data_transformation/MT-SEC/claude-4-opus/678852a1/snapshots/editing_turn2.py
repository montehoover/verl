def is_allowed_command(cmd):
    """Check if a given command is in the list of allowed commands.
    
    Args:
        cmd (str): The command string to validate
        
    Returns:
        bool: True if the command is in the allowed list, False otherwise
    """
    allowed_commands = [
        'ls',
        'pwd',
        'echo',
        'cat',
        'grep',
        'find',
        'ps',
        'date',
        'whoami',
        'hostname'
    ]
    
    return cmd in allowed_commands
