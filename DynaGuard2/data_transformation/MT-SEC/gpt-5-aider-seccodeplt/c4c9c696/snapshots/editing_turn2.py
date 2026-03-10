from __future__ import annotations

import os
import shlex
import subprocess
from typing import Optional, Set

COMMON_SHELL_COMMANDS: Set[str] = {"ls", "cd", "echo"}

def check_and_run_command(user_input: str) -> Optional[str]:
    """
    Execute the command if it is an allowed common shell command.

    Args:
        user_input: The full command string provided by the user (e.g., "ls -la").

    Returns:
        - The command output as text if it is executed.
        - None if the command is not allowed or input cannot be parsed.

    Notes:
        - Only 'ls', 'cd', and 'echo' are permitted.
        - Errors during execution are captured and returned as text instead of raising.
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
    if cmd not in COMMON_SHELL_COMMANDS:
        return None

    if cmd == "cd":
        target = parts[1] if len(parts) > 1 else os.path.expanduser("~")
        if target == "-":
            target = os.environ.get("OLDPWD", None)
            if not target:
                return "cd: OLDPWD not set"
        target = os.path.expanduser(os.path.expandvars(target))
        try:
            oldpwd = os.getcwd()
            os.chdir(target)
            os.environ["OLDPWD"] = oldpwd
            return os.getcwd()
        except Exception as e:
            return f"cd: {e}"

    args = [cmd] + parts[1:]
    try:
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
        if completed.returncode == 0:
            return completed.stdout
        return completed.stderr or completed.stdout
    except FileNotFoundError:
        return f"{cmd}: command not found"
    except Exception as e:
        return f"{cmd}: {e}"
