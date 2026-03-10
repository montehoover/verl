from typing import Optional

# Predefined list of valid commands that the user is permitted to run.
ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def verify_and_run_command(command: str) -> Optional[str]:
    """
    Validate a user-issued command against the set of allowed commands.

    Args:
        command: A string representing the user's input command.

    Returns:
        The input command string if it is allowed; otherwise, None.
    """
    if not isinstance(command, str):
        return None

    return command if command in ALLOWED_COMMANDS else None
