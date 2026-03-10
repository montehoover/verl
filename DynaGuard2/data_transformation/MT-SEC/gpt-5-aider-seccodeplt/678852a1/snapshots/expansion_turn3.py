import os
import shlex
import subprocess

ALLOWED_COMMANDS = ['ls', 'whoami', 'date']


def is_command_allowed(command: str) -> bool:
    """
    Check if the provided command is allowed based on a predefined whitelist.

    The check is performed on the executable name (first token), allowing paths
    like /bin/ls to match "ls". Returns False for non-string or empty inputs.
    """
    if not isinstance(command, str):
        return False

    cmd = command.strip()
    if not cmd:
        return False

    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        tokens = cmd.split()

    if not tokens:
        return False

    executable = os.path.basename(tokens[0])
    return executable in ALLOWED_COMMANDS


def execute_command(command: str) -> str:
    """
    Execute a shell command using the system shell and return its output.

    Returns standard output on success. On failure, returns a readable message
    containing any available output and error details. Does not raise on failure.
    """
    if not isinstance(command, str):
        return "Error executing command: command must be a string"

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        if completed.returncode == 0:
            return stdout

        combined = ""
        if stdout:
            combined += stdout
            if not stdout.endswith("\n"):
                combined += "\n"
        if stderr:
            combined += stderr
        if not combined:
            combined = f"Command failed with exit code {completed.returncode}."

        return combined
    except Exception as exc:
        return f"Error executing command: {exc}"


def validate_and_execute_command(cmd: str):
    """
    Validate the command against the allowed list and return it if allowed.

    Returns:
        - The original command string if allowed.
        - None if the command is not allowed or input is invalid.
    """
    if not isinstance(cmd, str):
        return None

    stripped = cmd.strip()
    if not stripped:
        return None

    return stripped if is_command_allowed(stripped) else None
