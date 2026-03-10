"""Module for validating and executing system commands safely."""

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
    # Check if the command is in the allowed list
    if cmd in ALLOWED_COMMANDS:
        return cmd
    else:
        return None
