from __future__ import annotations

import os
import shlex
from typing import Set

COMMON_SHELL_COMMANDS: Set[str] = {"ls", "cd", "echo"}

def check_and_run_command(user_input: str) -> bool:
    """
    Check whether the given user_input starts with a common shell command.

    Args:
        user_input: The full command string provided by the user (e.g., "ls -la").

    Returns:
        True if the command (first token) is a common shell command, otherwise False.
    """
    if not user_input or not isinstance(user_input, str):
        return False

    try:
        parts = shlex.split(user_input, posix=True)
    except ValueError:
        parts = user_input.strip().split()

    if not parts:
        return False

    cmd = os.path.basename(parts[0])
    return cmd in COMMON_SHELL_COMMANDS
