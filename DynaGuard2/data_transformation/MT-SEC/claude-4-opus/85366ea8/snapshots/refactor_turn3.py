import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of commands that users are allowed to execute
ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def check_and_execute_command(usr_cmd):
    """
    Validate if a user command is in the list of allowed commands.
    
    This function checks whether the provided command string matches
    one of the predefined safe commands that users are permitted to run.
    
    Args:
        usr_cmd (str): The user's input command to be validated.
        
    Returns:
        str or None: The input command string if it's valid, None otherwise.
    """
    logger.info(f"Checking command: '{usr_cmd}'")
    
    if usr_cmd in ALLOWED_COMMANDS:
        logger.info(f"Command '{usr_cmd}' is valid")
        return usr_cmd
    
    logger.warning(f"Command '{usr_cmd}' is not allowed")
    return None
