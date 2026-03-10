import subprocess
import shlex

def execute_command(base_command: str, parameters: list) -> str:
    """
    Execute a shell command with the given parameters.
    
    Args:
        base_command: The main command to execute
        parameters: List of arguments to pass to the command
        
    Returns:
        The output produced by the executed shell command
        
    Raises:
        ValueError: If the command is incorrect or cannot be processed
    """
    try:
        # Properly quote the base command
        quoted_command = shlex.quote(base_command)
        
        # Properly quote each parameter
        quoted_params = [shlex.quote(str(param)) for param in parameters]
        
        # Combine command and parameters
        full_command = [base_command] + parameters
        
        # Execute the command
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Return the output, stripping any trailing newline
        return result.stdout.rstrip('\n')
        
    except subprocess.CalledProcessError as e:
        # Command failed with non-zero exit code
        raise ValueError(f"Command execution failed: {e.stderr or str(e)}")
    except FileNotFoundError:
        # Command not found
        raise ValueError(f"Command not found: {base_command}")
    except Exception as e:
        # Any other error
        raise ValueError(f"Failed to execute command: {str(e)}")
