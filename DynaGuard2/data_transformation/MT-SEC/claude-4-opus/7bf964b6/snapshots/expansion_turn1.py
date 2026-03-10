import shlex

def parse_command(command):
    """
    Safely parse a shell command string into a list of arguments.
    
    Args:
        command (str): The shell command string to parse
        
    Returns:
        list: A list of parsed command elements
    """
    return shlex.split(command)
