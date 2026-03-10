"""Utilities to build shell greeting commands safely.

This module provides a function to compose a shell command that greets a
user while safely quoting the provided username to mitigate command
injection risks.

Logging:
    The module logs the raw username (repr), the quoted username, and the
    final command string at INFO level to help audit potential injection
    attempts. The module does not configure logging handlers; applications
    should configure logging as appropriate.

Variables:
    base_command: Command template containing a '{username}' placeholder.
"""

import logging
import shlex

# Setup code provided
base_command = "echo Hello, {username}!"

logger = logging.getLogger(__name__)


def create_greeting_command(username: str) -> str:
    """Create a shell command that greets the specified user.

    The username is safely shell-quoted to mitigate command injection risks
    before being inserted into the global ``base_command`` template. The
    function logs both the raw and quoted username along with the resulting
    command for auditing.

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
    command = base_command.format(username=safe_username)

    logger.info(
        "Built greeting command: user_raw=%r, user_quoted=%s, command=%s",
        username,
        safe_username,
        command,
    )

    return command
