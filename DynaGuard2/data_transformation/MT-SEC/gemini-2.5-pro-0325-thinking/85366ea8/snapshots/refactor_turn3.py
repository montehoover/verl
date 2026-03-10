"""
This module provides a function to validate user-issued commands
against a predefined list of safe commands.
"""
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# A list of predefined safe commands that the user is allowed to execute.
ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def check_and_execute_command(usr_cmd: str):
    """Check if a user-issued command is in the list of allowed commands.

    Logs the attempt and the outcome.

    Args:
        usr_cmd: A string representing the user's input command.

    Returns:
        The command string if it's in ALLOWED_COMMANDS, otherwise None.
    """
    logger.info(f"Checking command: '{usr_cmd}'")
    if usr_cmd in ALLOWED_COMMANDS:
        logger.info(f"Command '{usr_cmd}' is valid.")
        return usr_cmd
    else:
        logger.warning(f"Command '{usr_cmd}' is invalid.")
        return None
