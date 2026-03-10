# Predefined list of commands
ALLOWED_COMMANDS = [
    "start",
    "stop",
    "restart",
    "status",
]

def is_command_valid(command: str) -> bool:
    """
    Return True if the given command exists in the predefined list of commands.
    Comparison is exact and case-sensitive.
    """
    if not isinstance(command, str):
        return False
    return command in ALLOWED_COMMANDS
