import shlex
import subprocess

def build_command_string(base_command, arguments):
    """
    Build a safe command string from a base command and list of arguments.
    
    Args:
        base_command (str): The main command to execute
        arguments (list): List of arguments for the command
        
    Returns:
        str: Safely concatenated command string
    """
    # Ensure base_command is a string
    if not isinstance(base_command, str):
        raise TypeError("base_command must be a string")
    
    # Ensure arguments is a list
    if not isinstance(arguments, list):
        raise TypeError("arguments must be a list")
    
    # Quote the base command
    safe_command = shlex.quote(base_command)
    
    # Quote each argument and build the command string
    safe_args = [shlex.quote(str(arg)) for arg in arguments]
    
    # Join command and arguments with spaces
    return ' '.join([safe_command] + safe_args)


def execute_and_handle_errors(command_string):
    """
    Execute a command string and handle any errors that may occur.
    
    Args:
        command_string (str): The command string to execute
        
    Returns:
        str: Command output on success, or error message on failure
    """
    try:
        # Execute the command
        result = subprocess.run(
            command_string,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Return stdout output
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        # Handle non-zero exit status
        error_msg = f"Command failed with exit code {e.returncode}"
        if e.stderr:
            error_msg += f": {e.stderr}"
        return error_msg
        
    except subprocess.TimeoutExpired as e:
        # Handle timeout
        return f"Command timed out: {str(e)}"
        
    except FileNotFoundError:
        # Handle command not found
        return "Command not found"
        
    except Exception as e:
        # Handle any other exceptions
        return f"Error executing command: {str(e)}"


def run_shell_command(cmd, arguments):
    """
    Execute a shell command with arguments and return the output.
    
    Args:
        cmd (str): The base command to execute
        arguments (list): List of string arguments for the command
        
    Returns:
        str: The output from the command execution
        
    Raises:
        ValueError: If the command is invalid or fails
    """
    # Validate input types
    if not isinstance(cmd, str):
        raise ValueError("cmd must be a string")
    
    if not isinstance(arguments, list):
        raise ValueError("arguments must be a list")
    
    # Validate that all arguments are strings
    for arg in arguments:
        if not isinstance(arg, str):
            raise ValueError("All arguments must be strings")
    
    # Validate that command is not empty
    if not cmd.strip():
        raise ValueError("Command cannot be empty")
    
    try:
        # Build command list for subprocess
        command_list = [cmd] + arguments
        
        # Execute the command
        result = subprocess.run(
            command_list,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Return the stdout output
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        # Raise ValueError for command failures
        error_msg = f"Command '{cmd}' failed with exit code {e.returncode}"
        if e.stderr:
            error_msg += f": {e.stderr}"
        raise ValueError(error_msg)
        
    except FileNotFoundError:
        # Raise ValueError for invalid commands
        raise ValueError(f"Command '{cmd}' not found")
        
    except Exception as e:
        # Raise ValueError for any other errors
        raise ValueError(f"Error executing command '{cmd}': {str(e)}")
