from typing import Optional

ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def verify_and_run_command(command: str) -> Optional[str]:
    """
    Verifies if a command is in a predefined set of safe commands.

    Args:
        command: The command string to check.

    Returns:
        The command if it's valid, or None if it does not.
    """
    if command in ALLOWED_COMMANDS:
        return command
    else:
        return None
