"""Module for validating and executing system commands safely."""

import logging

# List of commands that are allowed to be executed
ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def validate_and_execute_command(cmd: str) -> str:
    """
    Validate a user-provided command against a whitelist of allowed commands.
    
    This function checks if the provided command is in the list of allowed
    commands to prevent arbitrary command execution.
    
    Args:
        cmd (str): The user-provided command to be validated.
        
    Returns:
        str: The command string if it's in the allowed list, None otherwise.
    """
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Log the validation attempt
    logger.info(f"Validating command: '{cmd}'")
    
    # Check if the command is in the allowed list
    if cmd in ALLOWED_COMMANDS:
        logger.info(f"Command '{cmd}' validated successfully")
        return cmd
    else:
        logger.warning(f"Command '{cmd}' rejected - not in allowed list")
        return None
