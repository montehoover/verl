def check_and_run_command(user_input: str) -> bool:
    """
    Check if a given command is one of the common shell commands.
    
    Args:
        user_input: The command string provided by the user
        
    Returns:
        True if the command is a common shell command, otherwise False
    """
    common_commands = {'ls', 'cd', 'echo'}
    
    # Extract the base command (first word)
    command = user_input.strip().split()[0] if user_input.strip() else ''
    
    return command in common_commands
