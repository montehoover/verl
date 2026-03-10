import subprocess
import shlex

def run_shell_command(cmd: str, arguments: list) -> str:
    """
    Run a shell command with specified arguments.
    
    Args:
        cmd: The primary command to run
        arguments: A list containing arguments for the command
        
    Returns:
        The result output from the executed command
        
    Raises:
        ValueError: Raised in case the command is invalid or fails to execute
    """
    try:
        # Safely quote the command
        safe_cmd = shlex.quote(cmd)
        
        # Safely quote each argument
        safe_args = [shlex.quote(arg) for arg in arguments]
        
        # Build the full command
        full_command = [cmd] + arguments
        
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
        raise ValueError(f"Command '{cmd}' not found")
    except Exception as e:
        raise ValueError(f"Failed to execute command: {str(e)}")
