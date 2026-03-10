import shlex
import os

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

def filter_and_run_command(cmd_input, permitted_subcommands):
    """
    Validate and execute a shell command if it passes the permitted subcommands filter.
    
    Args:
        cmd_input (str): The command string to validate and execute
        permitted_subcommands (list): List of permitted subcommands
        
    Returns:
        str: The output of the successful command or an error message if not allowed
        
    Raises:
        ValueError: If the command is invalid
    """
    command_components = parse_command(cmd_input)
    
    if not validate_command(command_components, permitted_subcommands):
        raise ValueError(f"Invalid command: '{cmd_input}' contains disallowed subcommands")
    
    result = os.popen(cmd_input).read()
    return result
