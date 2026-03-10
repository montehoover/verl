import shlex

base_command = "echo Hello, {username}!"

def create_greeting_command(username: str) -> str:
    """
    Prepares a shell command to greet a user by inserting their
    provided name into a predefined command string.

    Args:
        username: The user's name to include in the command.

    Returns:
        A string representing the final shell command that safely
        incorporates the provided user name.
    """
    # Sanitize the username to prevent command injection
    safe_username = shlex.quote(username)
    return base_command.format(username=safe_username)
