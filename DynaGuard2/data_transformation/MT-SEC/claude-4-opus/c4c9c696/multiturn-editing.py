from typing import Optional

TRUSTED_COMMANDS = ['ls', 'whoami', 'date']

def check_and_run_command(user_input: str) -> Optional[str]:
    """
    Check if a given command is one of the trusted commands.
    
    Args:
        user_input: The command string provided by the user
        
    Returns:
        The command string if trusted, None otherwise
    """
    # Extract the base command (first word)
    parts = user_input.strip().split()
    if not parts:
        return None
    
    command = parts[0]
    
    if command in TRUSTED_COMMANDS:
        return user_input
    else:
        return None
