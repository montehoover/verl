import logging

# List of commands that are allowed to be executed
ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def verify_and_run_command(command):
    """
    Verify if a user-issued command is within the predefined set of safe commands.
    
    This function checks whether the provided command string matches one of the
    commands in the ALLOWED_COMMANDS list. Only commands that are explicitly
    allowed will be validated for execution.
    
    Args:
        command (str): A string representing the user's input command to validate.
    
    Returns:
        str or None: The input command string if it matches an allowed command,
                     or None if the command is not in the allowed list.
    
    Examples:
        >>> verify_and_run_command('ls')
        'ls'
        >>> verify_and_run_command('rm')
        None
    """
    # Initialize logger for tracking command verification attempts
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create console handler if logger doesn't have handlers already
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter for clear and informative log messages
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
    
    # Log the command verification attempt
    logger.info(f"Command verification attempt: '{command}'")
    
    # Check if the command is in the allowed commands list
    if command in ALLOWED_COMMANDS:
        logger.info(f"Command '{command}' is ALLOWED - verification successful")
        return command
    else:
        logger.warning(f"Command '{command}' is NOT ALLOWED - verification failed")
        return None
