import shlex

def parse_command(command_string):
    """
    Safely parse a command string into its components.
    
    Args:
        command_string (str): The command string to parse
        
    Returns:
        list: A list of command components
    """
    return shlex.split(command_string)

def validate_command(command_components, permitted_subcommands):
    """
    Validate that all command components are in the permitted list.
    
    Args:
        command_components (list): List of command components to validate
        permitted_subcommands (list): List of permitted subcommands
        
    Returns:
        bool: True if all components are permitted, False otherwise
    """
    for component in command_components:
        if component not in permitted_subcommands:
            return False
    return True
