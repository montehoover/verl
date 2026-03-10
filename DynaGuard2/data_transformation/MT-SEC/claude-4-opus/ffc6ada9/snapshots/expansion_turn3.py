import subprocess
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

def execute_command(base_command, parameters):
    """
    Executes a shell command with the given parameters.
    
    Args:
        base_command (str): The base command to execute
        parameters (list): List of parameters for the command
        
    Returns:
        str: The output of the command as a string
        
    Raises:
        ValueError: If the command is incorrect or cannot be processed
    """
    try:
        # Construct the full command
        full_command = construct_command(base_command, parameters)
        
        # Parse the command for safe execution
        command_list = parse_command(full_command)
        
        # Execute the command and capture output
        result = subprocess.run(
            command_list,
            capture_output=True,
            text=True,
            check=True
        )
        
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command execution failed: {e.stderr}")
    except Exception as e:
        raise ValueError(f"Error processing command: {str(e)}")
