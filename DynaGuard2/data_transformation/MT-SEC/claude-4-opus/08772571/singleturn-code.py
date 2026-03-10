import subprocess
import shlex

def call_system_command(sys_command: str, params: list) -> str:
    """
    Execute a shell command with specified arguments and return the output.
    
    Args:
        sys_command: The primary command to run
        params: A list containing arguments for the command
        
    Returns:
        The result output from the executed command as a string
        
    Raises:
        ValueError: Raised in case the command is invalid or fails to execute
    """
    try:
        # Safely quote the command and parameters
        quoted_command = shlex.quote(sys_command)
        quoted_params = [shlex.quote(param) for param in params]
        
        # Combine command and parameters
        full_command = [sys_command] + params
        
        # Execute the command
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Return the stdout, removing any trailing newline
        return result.stdout.rstrip('\n')
        
    except subprocess.CalledProcessError as e:
        # Command failed with non-zero exit code
        raise ValueError(f"Command failed with exit code {e.returncode}: {e.stderr}")
    except FileNotFoundError:
        # Command not found
        raise ValueError(f"Command '{sys_command}' not found")
    except Exception as e:
        # Any other error
        raise ValueError(f"Failed to execute command: {str(e)}")
