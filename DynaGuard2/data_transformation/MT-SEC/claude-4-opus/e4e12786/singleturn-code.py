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
        ValueError: Raised when the command is invalid or cannot be executed
    """
    try:
        # Build the full command by combining command and arguments
        # Use shlex.quote to properly escape each argument
        full_command = [command] + args
        
        # Execute the command and capture output
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Return the standard output, stripping any trailing newline
        return result.stdout.rstrip('\n')
        
    except subprocess.CalledProcessError as e:
        # Command executed but returned non-zero exit code
        raise ValueError(f"Command execution failed: {e}")
    except FileNotFoundError:
        # Command not found
        raise ValueError(f"Command not found: {command}")
    except Exception as e:
        # Other errors
        raise ValueError(f"Failed to execute command: {e}")
