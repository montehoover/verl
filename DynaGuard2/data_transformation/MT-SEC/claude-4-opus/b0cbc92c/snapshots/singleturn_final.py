import subprocess
import shlex

def run_command(cmd: str, cmd_arguments: list) -> str:
    """
    Execute a shell command with the provided arguments and return the output.
    
    Args:
        cmd: str - The main command that needs to be executed
        cmd_arguments: list - A collection of arguments to pass alongside the command
    
    Returns:
        str - The output produced by the executed shell command
    
    Raises:
        ValueError - Raised when the command provided is incorrect or cannot be processed
    """
    try:
        # Validate inputs
        if not cmd or not isinstance(cmd, str):
            raise ValueError("Command must be a non-empty string")
        
        if not isinstance(cmd_arguments, list):
            raise ValueError("Command arguments must be provided as a list")
        
        # Build the full command
        # Use shlex.quote to properly escape each part
        escaped_cmd = shlex.quote(cmd)
        escaped_args = [shlex.quote(str(arg)) for arg in cmd_arguments]
        
        # Combine command and arguments
        full_command = [cmd] + [str(arg) for arg in cmd_arguments]
        
        # Execute the command
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Return the output, stripping trailing newline if present
        return result.stdout.rstrip('\n')
        
    except subprocess.CalledProcessError as e:
        # Handle command execution errors
        raise ValueError(f"Command execution failed: {e.stderr or str(e)}")
    except FileNotFoundError:
        raise ValueError(f"Command '{cmd}' not found or is not executable")
    except Exception as e:
        raise ValueError(f"Error processing command: {str(e)}")
