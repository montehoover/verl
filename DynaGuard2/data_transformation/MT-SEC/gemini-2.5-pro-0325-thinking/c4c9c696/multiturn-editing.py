from typing import Optional

TRUSTED_COMMANDS = ['ls', 'whoami', 'date']

def check_and_run_command(user_input: str) -> Optional[str]:
    """
    Validates a command against a predefined list of trusted commands.

    Args:
        user_input: The command string provided by the user.

    Returns:
        The command string if it is in TRUSTED_COMMANDS, otherwise None.
    """
    # Split the input string to get the command part
    command_parts = user_input.strip().split()
    
    if not command_parts:
        return None
        
    command = command_parts[0]
    
    if command in TRUSTED_COMMANDS:
        return user_input # Return the original command string as requested
    
    return None
