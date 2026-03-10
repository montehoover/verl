import shlex

def construct_command(base_command, arguments):
    """
    Safely construct a shell command string from a base command and arguments.
    
    Args:
        base_command: The base command as a string
        arguments: List of arguments to append to the command
        
    Returns:
        A properly escaped shell command string
    """
    # Start with the base command
    command_parts = [base_command]
    
    # Add each argument, properly quoted for shell safety
    for arg in arguments:
        command_parts.append(shlex.quote(str(arg)))
    
    # Join all parts with spaces
    return ' '.join(command_parts)
