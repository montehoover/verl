import shlex

# Provided command template with a placeholder for the username.
# This can be overridden by the environment importing this module, if desired.
base_command: str = "echo Hello, {username}!"

def create_greeting_command(username: str) -> str:
    """
    Prepare a shell command to greet a user by safely inserting their name into a predefined command string.

    Args:
        username (str): The user's name to include in the command.

    Returns:
        str: The final shell command that safely incorporates the provided user name.
    """
    # Ensure the input is a string (defensive)
    if not isinstance(username, str):
        username = str(username)

    # Sanitize to prevent command injection by shell-quoting the username
    safe_username = shlex.quote(username)

    # base_command is expected to be provided externally, e.g.:
    # base_command = "echo Hello, {username}!"
    return base_command.format(username=safe_username)
