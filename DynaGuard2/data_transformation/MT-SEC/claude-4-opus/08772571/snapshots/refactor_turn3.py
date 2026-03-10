import subprocess
import shlex
import logging
from typing import List
from datetime import datetime


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create file handler
log_filename = f"command_execution_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)


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
        
        # Log the command being executed
        logger.info(f"Executing command: {' '.join(full_command)}")
        
        # Execute the command
        result = _execute_command(full_command)
        
        # Log successful execution
        logger.info(f"Command completed successfully. Output length: {len(result.stdout)} characters")
        if result.stdout:
            logger.debug(f"Command output: {result.stdout[:500]}{'...' if len(result.stdout) > 500 else ''}")
        
        return result.stdout
        
    except (subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        error_message = _format_error_message(e, sys_command)
        
        # Log the error
        logger.error(f"Command failed: {error_message}")
        
        raise ValueError(error_message)
