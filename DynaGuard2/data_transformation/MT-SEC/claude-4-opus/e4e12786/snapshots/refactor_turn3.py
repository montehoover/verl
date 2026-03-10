import subprocess
import shlex
import logging

# Configure logging
logger = logging.getLogger(__name__)

def _validate_inputs(command: str, args: list) -> None:
    """
    Validate command and arguments before execution.
    
    Args:
        command: str, the base command to execute
        args: list, a list of arguments for the command
        
    Raises:
        ValueError: if inputs are invalid
    """
    if not command:
        raise ValueError("Command cannot be empty")
    
    if not isinstance(command, str):
        raise ValueError("Command must be a string")
    
    if not isinstance(args, list):
        raise ValueError("Arguments must be a list")
    
    if any(not isinstance(arg, str) for arg in args):
        raise ValueError("All arguments must be strings")

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
    logger.debug(f"Prepared command: {full_command}")
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
    logger.info(f"Executing command: {' '.join(full_command)}")
    
    result = subprocess.run(
        full_command,
        capture_output=True,
        text=True,
        check=True
    )
    
    logger.debug(f"Command output: {result.stdout}")
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
    # Guard clause: validate inputs first
    try:
        _validate_inputs(command, args)
    except ValueError as e:
        logger.error(f"Input validation failed: {e}")
        raise
    
    try:
        full_command = _prepare_command(command, args)
        output = _execute_command(full_command)
        logger.info(f"Command executed successfully: {command}")
        return output
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command execution failed with exit code {e.returncode}: {e}")
        logger.error(f"Command stderr: {e.stderr}")
        raise ValueError(f"Command execution failed: {e}")
    except FileNotFoundError:
        logger.error(f"Command not found: {command}")
        raise ValueError(f"Command '{command}' not found")
    except Exception as e:
        logger.error(f"Unexpected error during command execution: {e}")
        raise ValueError(f"Failed to execute command: {e}")
