import subprocess
import shlex


def run_command(cmd: str, cmd_arguments: list) -> str:
    """
    Execute a shell command with arguments and return the output.
    
    This function safely executes a shell command with the provided arguments
    by properly handling command injection risks and capturing the output.
    The command and its arguments are processed separately to ensure safe
    execution.
    
    Args:
        cmd (str): The main command to execute. This should be the name
            or path of the executable program to run.
        cmd_arguments (list): A list of arguments to pass to the command.
            Each argument should be a separate element in the list.
            Arguments will be automatically converted to strings if needed.
            
    Returns:
        str: The standard output produced by the executed command.
            If the command produces no output, an empty string is returned.
            
    Raises:
        ValueError: Raised in the following cases:
            - When the command execution fails (non-zero exit code)
            - When the specified command cannot be found in the system
            - When any other error occurs during command processing
            
    Examples:
        >>> run_command("echo", ["Hello", "World"])
        'Hello World\\n'
        
        >>> run_command("ls", ["-la", "/tmp"])
        'total 16\\ndrwxrwxrwt  4 root root 4096 Oct 20 10:30 .\\n...'
        
        >>> run_command("nonexistent", [])
        Traceback (most recent call last):
            ...
        ValueError: Command 'nonexistent' not found
    """
    try:
        # Build the full command list with the base command and arguments
        full_command = [cmd] + cmd_arguments
        
        # Execute the command and capture output
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=True
        )
        
        return result.stdout
        
    except subprocess.CalledProcessError as e:
        # Handle non-zero exit codes
        error_message = f"Command execution failed: {e.stderr}"
        raise ValueError(error_message)
        
    except FileNotFoundError:
        # Handle missing commands
        error_message = f"Command '{cmd}' not found"
        raise ValueError(error_message)
        
    except Exception as e:
        # Handle any other unexpected errors
        error_message = f"Error processing command: {str(e)}"
        raise ValueError(error_message)
