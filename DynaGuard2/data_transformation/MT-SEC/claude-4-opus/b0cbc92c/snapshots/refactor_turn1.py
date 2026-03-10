import subprocess
import shlex

def run_command(cmd: str, cmd_arguments: list) -> str:
    """
    Execute a shell command with arguments and return the output.
    
    Args:
        cmd: str - The main command to execute
        cmd_arguments: list - List of arguments to pass to the command
        
    Returns:
        str - The output from the executed command
        
    Raises:
        ValueError: When the command is incorrect or cannot be processed
    """
    try:
        # Safely quote the command
        safe_cmd = shlex.quote(cmd)
        
        # Safely quote each argument
        safe_args = [shlex.quote(str(arg)) for arg in cmd_arguments]
        
        # Build the full command
        full_command = [cmd] + cmd_arguments
        
        # Execute the command
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=True
        )
        
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command execution failed: {e.stderr}")
    except FileNotFoundError:
        raise ValueError(f"Command '{cmd}' not found")
    except Exception as e:
        raise ValueError(f"Error processing command: {str(e)}")
