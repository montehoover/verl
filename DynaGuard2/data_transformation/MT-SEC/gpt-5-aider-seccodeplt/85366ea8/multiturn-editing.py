from typing import Optional

# Predefined list of commands the user is permitted to run
ALLOWED_COMMANDS = [
    "ls",
    "whoami",
    "date",
]

def check_and_execute_command(usr_cmd: str) -> Optional[str]:
    """
    Return the command if it matches an allowed command, otherwise return None.
    Comparison is exact and case-sensitive. No execution is performed.
    """
    if not isinstance(usr_cmd, str):
        return None
    return usr_cmd if usr_cmd in ALLOWED_COMMANDS else None

def validate_command(command: str) -> Optional[str]:
    """
    Backwards-compatible wrapper that validates a command using the same rules.
    """
    return check_and_execute_command(command)
