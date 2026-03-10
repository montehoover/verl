import subprocess
import shlex

def _prepare_command(command: str, args: list) -> list:
    """
    Prepare a command with arguments for safe execution.
    
    Args:
        command: str, the base command to execute
        args: list, a list of arguments for the command
        
    Returns:
        list, the prepared command with arguments
    """
    # Combine command and arguments
    full_command = [command] + args
    return full_command

def _execute_command(full_command: list) -> str:
    """
    Execute a prepared command and return its output.
    
    Args:
        full_command: list, the command with arguments to execute
        
    Returns:
        str, the output of the executed command
        
    Raises:
        subprocess.CalledProcessError: if the command returns non-zero exit status
        FileNotFoundError: if the command is not found
    """
    result = subprocess.run(
        full_command,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout

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
        full_command = _prepare_command(command, args)
        output = _execute_command(full_command)
        return output
        
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Command execution failed: {e}")
    except FileNotFoundError:
        raise ValueError(f"Command '{command}' not found")
    except Exception as e:
        raise ValueError(f"Failed to execute command: {e}")
