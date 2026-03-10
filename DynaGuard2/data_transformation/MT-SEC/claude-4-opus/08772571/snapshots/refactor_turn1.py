import subprocess
import shlex

def call_system_command(sys_command: str, params: list) -> str:
    """
    Run a shell command with user-specified arguments.
    
    Args:
        sys_command: The primary command to run
        params: A list containing arguments for the command
        
    Returns:
        The result output from the executed command
        
    Raises:
        ValueError: Raised in case the command is invalid or fails to execute
    """
    try:
        # Safely quote the command
        safe_command = shlex.quote(sys_command)
        
        # Safely quote each parameter
        safe_params = [shlex.quote(str(param)) for param in params]
        
        # Build the full command
        full_command = [sys_command] + params
        
        # Execute the command
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=True
        )
        
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command failed with exit code {e.returncode}: {e.stderr}")
    except FileNotFoundError:
        raise ValueError(f"Command not found: {sys_command}")
    except Exception as e:
        raise ValueError(f"Failed to execute command: {str(e)}")
