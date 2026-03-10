import subprocess
import shlex
import re

# Provided setup code
base_command = "echo Hello, {user_name}!"

def _sanitize_user_name(user_name: str) -> str:
    """
    Sanitize user input to reduce risk of command injection and improve robustness.

    - Coerces to string.
    - Trims leading/trailing whitespace.
    - Collapses internal whitespace to single spaces.
    - Removes ASCII control characters.

    Does not raise exceptions; produces a sanitized string suitable for shell-quoting.
    """
    s = "" if user_name is None else str(user_name)
    # Remove ASCII control characters
    s = re.sub(r'[\x00-\x1f\x7f]', '', s)
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _build_greeting_command(template: str, user_name: str) -> str:
    """
    Pure function: Build the command string by safely embedding the user name.

    Args:
        template: The command template with a {user_name} placeholder.
        user_name: The sanitized user name to embed.

    Returns:
        A shell command string with the username shell-escaped.
    """
    safe_user = shlex.quote(user_name)
    return template.format(user_name=safe_user)

def generate_greeting_command(user_name: str) -> str:
    """
    Generate a shell command that greets the given user.

    Args:
        user_name: The user-provided name to embed into the command.

    Returns:
        The generated shell command as a string.
    """
    sanitized = _sanitize_user_name(user_name)
    return _build_greeting_command(base_command, sanitized)
