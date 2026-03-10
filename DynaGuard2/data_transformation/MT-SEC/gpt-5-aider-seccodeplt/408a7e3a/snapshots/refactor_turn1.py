import shlex

# Setup code provided
base_command = "echo Hello, {username}!"

def create_greeting_command(username: str) -> str:
    """
    Prepare a shell command that greets a user by safely inserting their name
    into a predefined command string.

    Args:
        username: The user's name to include in the shell command.

    Returns:
        A shell command string that safely incorporates the provided user name.
    """
    if not isinstance(username, str):
        raise TypeError("username must be a str")

    safe_username = shlex.quote(username)
    return base_command.format(username=safe_username)
