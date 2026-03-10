import subprocess
import shlex

def execute_shell_command(command: str, args: list) -> str:
    """
    Execute a shell command with user-provided arguments.
    
    Args:
        command: str, the base command to execute
        args: list, a list of arguments for the command
        
    Returns:
        str, the output of the executed command
        
    Raises:
        ValueError: raised when the command is invalid or cannot be executed
    """
    try:
        # Safely quote the command
        safe_command = shlex.quote(command)
        
        # Safely quote each argument
        safe_args = [shlex.quote(arg) for arg in args]
        
        # Combine command and arguments
        full_command = [command] + args
        
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
        raise ValueError(f"Command '{command}' not found")
    except Exception as e:
        raise ValueError(f"Failed to execute command: {e}")
