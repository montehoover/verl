import shlex

base_command = "echo Hello, {username}!"


def create_greeting_command(username: str) -> str:
    """Create a shell command to greet a user by name.

    This function takes a username and safely incorporates it into a
    predefined shell command template. The username is sanitized using
    shlex.quote() to prevent command injection attacks.

    Args:
        username: The user's name to be included in the greeting command.

    Returns:
        A string containing the complete shell command with the sanitized
        username inserted into the command template.

    Example:
        >>> create_greeting_command("Alice")
        "echo Hello, Alice!"
        >>> create_greeting_command("Alice; rm -rf /")
        "echo Hello, 'Alice; rm -rf /'!"
    """
    # Sanitize the username to prevent command injection
    safe_username = shlex.quote(username)
    # Format the command with the sanitized username
    return base_command.format(username=safe_username)
