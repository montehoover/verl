from typing import Optional

ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def is_command_allowed(command: str) -> str:
    """
    Return a message indicating whether the given command is in the allowed commands list.
    """
    if command in ALLOWED_COMMANDS:
        return 'Command is allowed'
    return 'Command is not allowed'

def verify_and_run_command(command: str) -> Optional[str]:
    """
    Return the command if it is in the allowed commands list; otherwise return None.
    """
    return command if command in ALLOWED_COMMANDS else None
