import os
import shlex
import subprocess
import logging
from datetime import datetime

# Configure logging
log_filename = os.path.join(os.getcwd(), f"command_filter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_command(cmd_input: str) -> list:
    """
    Parse the command string into a list of command parts.
    
    Args:
        cmd_input: str, the command to parse
    
    Returns:
        list, the parsed command parts
    
    Raises:
        ValueError: Raised when the command syntax is invalid
    """
    try:
        cmd_parts = shlex.split(cmd_input)
    except ValueError as e:
        raise ValueError(f"Invalid command syntax: {e}")
    
    if not cmd_parts:
        raise ValueError("Empty command provided")
    
    return cmd_parts

def validate_command_parts(cmd_parts: list, permitted_subcommands: list) -> None:
    """
    Validate that all command parts are in the permitted list.
    
    Args:
        cmd_parts: list, the command parts to validate
        permitted_subcommands: list, the list of permitted subcommands
    
    Raises:
        ValueError: Raised when a command part is not permitted
    """
    for part in cmd_parts:
        if part not in permitted_subcommands:
            raise ValueError(f"Command part '{part}' is not in the list of permitted subcommands")

def execute_command(cmd_parts: list) -> str:
    """
    Execute the command and return the output.
    
    Args:
        cmd_parts: list, the command parts to execute
    
    Returns:
        str, the output of the command or an error message
    """
    try:
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command failed with error: {e.stderr}"
    except Exception as e:
        return f"Command execution failed: {str(e)}"

def filter_and_run_command(cmd_input: str, permitted_subcommands: list) -> str:
    """
    Execute a shell command only if it passes through specific filters.
    
    Args:
        cmd_input: str, the command to execute
        permitted_subcommands: list, a list of subcommands that are considered safe and permitted
    
    Returns:
        str, the output of the successful command or an error message
    
    Raises:
        ValueError: Raised when the command or any of its subcommands do not match the allowed list
    """
    logger.info(f"Command execution attempt: '{cmd_input}'")
    logger.info(f"Permitted subcommands: {permitted_subcommands}")
    
    try:
        # Parse the command into parts
        cmd_parts = parse_command(cmd_input)
        logger.info(f"Parsed command parts: {cmd_parts}")
        
        # Validate command parts
        validate_command_parts(cmd_parts, permitted_subcommands)
        logger.info("Command validation: PASSED")
        
        # Execute the command
        result = execute_command(cmd_parts)
        logger.info(f"Command execution: SUCCESS")
        logger.info(f"Command output: {result}")
        return result
        
    except ValueError as e:
        logger.warning(f"Command validation: FAILED - {str(e)}")
        logger.warning(f"Command '{cmd_input}' was not allowed")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise
