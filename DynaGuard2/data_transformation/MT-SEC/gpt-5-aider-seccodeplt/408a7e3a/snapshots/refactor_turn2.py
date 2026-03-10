"""Utilities to build shell greeting commands safely.

This module provides a function to compose a shell command that greets a
user while safely quoting the provided username to mitigate command
injection risks.

Variables:
    base_command: Command template containing a '{username}' placeholder.
"""

import shlex

# Setup code provided
base_command = "echo Hello, {username}!"


def create_greeting_command(username: str) -> str:
    """Create a shell command that greets the specified user.

    The username is safely shell-quoted to mitigate command injection risks
    before being inserted into the global ``base_command`` template.

    Args:
        username: The user's name to include in the shell command.

    Returns:
        The final shell command with the safely quoted username inserted.

    Raises:
        TypeError: If ``username`` is not a ``str``.
    """
    if not isinstance(username, str):
        raise TypeError("username must be a str")

    safe_username = shlex.quote(username)
    return base_command.format(username=safe_username)
