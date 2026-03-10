import os
import shlex
import subprocess

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

def validate_and_execute_command(shell_cmd, safe_subcommands):
    """
    Parse, validate, and execute a shell command if it passes validation.
    
    Args:
        shell_cmd (str): The shell command string to execute
        safe_subcommands (list): List of permitted subcommands
        
    Returns:
        str: Command output if successful, error message otherwise
    """
    try:
        parsed_cmd = parse_command(shell_cmd)
        
        if not validate_subcommands(parsed_cmd, safe_subcommands):
            return "Error: Command contains unauthorized subcommands"
        
        result = subprocess.run(parsed_cmd, capture_output=True, text=True, check=True)
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"
