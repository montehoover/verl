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

def validate_subcommands(parsed_command, allowed_subcommands):
    """
    Validate that all elements in the parsed command are within the allowed set.
    
    Args:
        parsed_command (list): List of command elements from parse_command
        allowed_subcommands (list): List of permitted subcommands
        
    Returns:
        bool: True if all elements are allowed, False otherwise
    """
    return all(element in allowed_subcommands for element in parsed_command)
