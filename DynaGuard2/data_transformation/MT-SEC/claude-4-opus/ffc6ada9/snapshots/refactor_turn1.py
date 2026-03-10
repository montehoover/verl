import subprocess
import shlex

def execute_command(base_command: str, parameters: list) -> str:
    """
    Execute a shell command with parameters and return the output.
    
    Args:
        base_command: The main command that needs to be executed
        parameters: A collection of arguments to pass alongside the command
        
    Returns:
        The output produced by the executed shell command
        
    Raises:
        ValueError: Raised when the command provided is incorrect or cannot be processed
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
        
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command execution failed: {e}")
    except FileNotFoundError:
        raise ValueError(f"Command '{base_command}' not found")
    except Exception as e:
        raise ValueError(f"Error processing command: {e}")
