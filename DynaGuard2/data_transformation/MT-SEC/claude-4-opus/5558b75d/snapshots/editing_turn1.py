def validate_command(command, permitted_subcommands):
    """
    Check if a command string contains only allowed subcommands.
    
    Args:
        command: A string containing the command to validate
        permitted_subcommands: A list of allowed subcommand strings
        
    Returns:
        True if the command contains only permitted subcommands, False otherwise
    """
    # Split the command into individual parts
    command_parts = command.split()
    
    # Check each part of the command
    for part in command_parts:
        # Check if this part is in the permitted subcommands
        if part not in permitted_subcommands:
            return False
    
    return True
