import subprocess
import shlex
from typing import List


def _build_command(sys_command: str, params: List[str]) -> List[str]:
    """
    Build a command list from the base command and parameters.
    
    Args:
        sys_command: The primary command to run
        params: A list containing arguments for the command
        
    Returns:
        A list representing the full command
    """
    return [sys_command] + params


def _execute_command(command: List[str]) -> subprocess.CompletedProcess:
    """
    Execute a command and return the completed process.
    
    Args:
        command: The command list to execute
        
    Returns:
        The completed process object
        
    Raises:
        subprocess.CalledProcessError: If the command returns non-zero exit code
        FileNotFoundError: If the command is not found
    """
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )


def _format_error_message(error: Exception, sys_command: str) -> str:
    """
    Format an error message based on the exception type.
    
    Args:
        error: The exception that occurred
        sys_command: The command that was attempted
        
    Returns:
        A formatted error message
    """
    if isinstance(error, subprocess.CalledProcessError):
        return f"Command failed with exit code {error.returncode}: {error.stderr}"
    elif isinstance(error, FileNotFoundError):
        return f"Command not found: {sys_command}"
    else:
        return f"Failed to execute command: {str(error)}"


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
        # Build the full command
        full_command = _build_command(sys_command, params)
        
        # Execute the command
        result = _execute_command(full_command)
        
        return result.stdout
        
    except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        error_message = _format_error_message(e, sys_command)
        raise ValueError(error_message)
