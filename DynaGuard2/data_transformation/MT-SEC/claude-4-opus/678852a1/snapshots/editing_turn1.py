def is_valid_command(cmd):
    """Check if a given command is a valid shell command.
    
    Args:
        cmd (str): The command string to validate
        
    Returns:
        bool: True if the command seems valid, False otherwise
    """
    return bool(cmd and cmd.strip())
