from typing import Optional

def validate_command(command: str) -> Optional[str]:
    """
    Checks if the given command is in a predefined list of commands.
    If valid, returns the command string. Otherwise, returns None.

    Args:
        command: The command string to validate.

    Returns:
        The command string if it is valid, None otherwise.
    """
    valid_commands = ["start", "stop", "status", "restart"]  # Example list of commands
    if command in valid_commands:
        return command
    return None
