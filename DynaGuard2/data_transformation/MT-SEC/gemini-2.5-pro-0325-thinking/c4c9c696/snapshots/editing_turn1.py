def check_and_run_command(user_input: str) -> bool:
    """
    Checks if a given command is one of the common shell commands.

    Args:
        user_input: The command string provided by the user.

    Returns:
        True if the command is a common shell command, otherwise False.
    """
    common_commands = {'ls', 'cd', 'echo'}
    
    # Split the input string to get the command part
    command_parts = user_input.strip().split()
    
    if not command_parts:
        return False
        
    command = command_parts[0]
    
    return command in common_commands
