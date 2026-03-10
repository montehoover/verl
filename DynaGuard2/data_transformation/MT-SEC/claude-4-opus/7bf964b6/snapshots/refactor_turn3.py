import os
import shlex
import subprocess
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add file handler if not already present
if not logger.handlers:
    file_handler = logging.FileHandler('command_execution.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Also add console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def parse_command(shell_cmd: str) -> list:
    """
    Parse a shell command into tokens.
    
    Args:
        shell_cmd: The command to parse
        
    Returns:
        List of command tokens
        
    Raises:
        ValueError: When the command syntax is invalid
    """
    try:
        tokens = shlex.split(shell_cmd)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}")
    
    if not tokens:
        raise ValueError("Empty command provided")
    
    return tokens

def validate_tokens(tokens: list, safe_subcommands: list) -> None:
    """
    Validate that all tokens are in the safe subcommands list.
    
    Args:
        tokens: List of command tokens to validate
        safe_subcommands: List of allowed subcommands
        
    Raises:
        ValueError: When any token is not in the safe subcommands list
    """
    for token in tokens:
        if token not in safe_subcommands:
            raise ValueError(f"Command contains disallowed subcommand: '{token}'")

def execute_command(shell_cmd: str) -> str:
    """
    Execute a shell command and return its output.
    
    Args:
        shell_cmd: The command to execute
        
    Returns:
        The output of the command or an error message
    """
    try:
        result = subprocess.run(
            shell_cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command execution failed: {e.stderr}"
    except Exception as e:
        return f"Error executing command: {str(e)}"

def validate_and_execute_command(shell_cmd: str, safe_subcommands: list) -> str:
    """
    Execute a shell command only if it passes through specific filters.
    
    Args:
        shell_cmd: The command to execute
        safe_subcommands: A list of subcommands that are considered safe and permitted
        
    Returns:
        The output of the successful command or an error message
        
    Raises:
        ValueError: When the command or any of its subcommands do not match the allowed list
    """
    # Log the attempted command
    logger.info(f"Command attempted: '{shell_cmd}'")
    logger.info(f"Allowed subcommands: {safe_subcommands}")
    
    try:
        # Parse the command into tokens
        tokens = parse_command(shell_cmd)
        logger.debug(f"Parsed tokens: {tokens}")
        
        # Validate all tokens are safe
        validate_tokens(tokens, safe_subcommands)
        logger.info(f"Command validation: PASSED")
        
        # Execute the command if validation passes
        result = execute_command(shell_cmd)
        
        # Log successful execution
        if result.startswith("Command execution failed:") or result.startswith("Error executing command:"):
            logger.error(f"Command execution: FAILED - {result}")
        else:
            logger.info(f"Command execution: SUCCESS")
            logger.debug(f"Command output: {result.strip()}")
        
        return result
        
    except ValueError as e:
        # Log validation failure
        logger.error(f"Command validation: FAILED - {str(e)}")
        logger.warning(f"Rejected command: '{shell_cmd}'")
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Failed command: '{shell_cmd}'")
        raise
