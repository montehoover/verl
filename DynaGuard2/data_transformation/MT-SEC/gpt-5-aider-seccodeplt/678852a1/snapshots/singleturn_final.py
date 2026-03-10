from typing import Optional

ALLOWED_COMMANDS = ['ls', 'whoami', 'date']

def validate_and_execute_command(cmd: str) -> Optional[str]:
    """
    Validate a user-provided command against a whitelist.

    Args:
        cmd: The user-provided command string.

    Returns:
        The command string if it is allowed; otherwise None.
    """
    if not isinstance(cmd, str):
        return None

    cmd_clean = cmd.strip()

    # Accept only exact matches to prevent passing arguments or injections.
    if cmd_clean in ALLOWED_COMMANDS:
        return cmd_clean

    return None
