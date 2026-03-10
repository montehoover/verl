def validate_command(command, permitted_subcommands):
    """
    Check if a command string contains only allowed subcommands and execute if valid.
    
    Args:
        command: A string containing the command to validate
        permitted_subcommands: A list of allowed subcommand strings
        
    Returns:
        The output of the command execution if valid, or a message indicating 
        the command is not permitted
    """
    # Split the command into individual parts
    command_parts = command.split()
    
    # Check each part of the command
    for part in command_parts:
        # Check if this part is in the permitted subcommands
        if part not in permitted_subcommands:
            return "Command not permitted"
    
    # Execute the command if it's valid
    try:
        # Using eval to execute the command string
        # Note: This is simplified and should be used with caution
        result = eval(command)
        return result
    except Exception as e:
        return f"Error executing command: {str(e)}"
