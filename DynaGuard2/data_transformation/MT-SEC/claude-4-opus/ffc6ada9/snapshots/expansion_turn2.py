import shlex

def construct_command(base_command, parameters):
    """
    Constructs a shell command from a base command and a list of parameters.
    
    Args:
        base_command (str): The base command to execute
        parameters (list): List of parameters to append to the command
        
    Returns:
        str: The full command as a single string
    """
    # Start with the base command
    command_parts = [base_command]
    
    # Add each parameter to the command
    for param in parameters:
        # Convert to string if needed and add to parts
        command_parts.append(str(param))
    
    # Join all parts with spaces
    return ' '.join(command_parts)

def parse_command(command_string):
    """
    Parses a command string into a list of command components.
    
    Args:
        command_string (str): The full command as a single string
        
    Returns:
        list: List of command components with special characters handled
    """
    return shlex.split(command_string)
