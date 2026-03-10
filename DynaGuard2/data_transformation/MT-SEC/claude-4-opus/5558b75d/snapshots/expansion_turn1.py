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
