import shlex
import subprocess

def construct_command(main_command, arguments):
    """
    Constructs a shell command string from a main command and a list of arguments.
    
    Args:
        main_command (str): The main command to execute
        arguments (list): List of arguments to pass to the command
        
    Returns:
        str: The complete command string with properly escaped arguments
    """
    # Start with the main command
    command_parts = [shlex.quote(main_command)]
    
    # Add each argument, properly quoted
    for arg in arguments:
        command_parts.append(shlex.quote(str(arg)))
    
    # Join all parts with spaces
    return ' '.join(command_parts)

def parse_command(command_string):
    """
    Parses a command string into a list of command components.
    
    Args:
        command_string (str): The command string to parse
        
    Returns:
        list: List of command components with proper quote handling
    """
    return shlex.split(command_string)

def run_command(cmd, cmd_arguments):
    """
    Executes a shell command with the given arguments and returns the output.
    
    Args:
        cmd (str): The main command to execute
        cmd_arguments (list): List of arguments to pass to the command
        
    Returns:
        str: The output of the command as a string
        
    Raises:
        ValueError: If the command is incorrect or cannot be processed
    """
    try:
        # Construct the full command list
        command_list = [cmd] + cmd_arguments
        
        # Execute the command and capture output
        result = subprocess.run(
            command_list,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Return the stdout output
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command execution failed: {e}")
    except Exception as e:
        raise ValueError(f"Error processing command: {e}")
