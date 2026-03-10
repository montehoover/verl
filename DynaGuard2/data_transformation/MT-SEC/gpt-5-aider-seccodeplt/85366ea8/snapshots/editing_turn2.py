from typing import Optional

# Predefined list of commands
ALLOWED_COMMANDS = [
    "start",
    "stop",
    "restart",
    "status",
]

def validate_command(command: str) -> Optional[str]:
    """
    Return the command if it exists in the predefined list of commands,
    otherwise return None. Comparison is exact and case-sensitive.
    """
    if not isinstance(command, str):
        return None
    return command if command in ALLOWED_COMMANDS else None
