from __future__ import annotations

import os
import shlex
from typing import Optional

TRUSTED_COMMANDS = ['ls', 'whoami', 'date']

def check_and_run_command(user_input: str) -> Optional[str]:
    """
    Validate the given user_input against TRUSTED_COMMANDS.
    Returns the original command string if the first token is trusted; otherwise None.
    """
    if not isinstance(user_input, str) or not user_input.strip():
        return None

    try:
        parts = shlex.split(user_input, posix=True)
    except ValueError:
        parts = user_input.strip().split()

    if not parts:
        return None

    cmd = os.path.basename(parts[0])
    if cmd in TRUSTED_COMMANDS:
        return user_input
    return None
