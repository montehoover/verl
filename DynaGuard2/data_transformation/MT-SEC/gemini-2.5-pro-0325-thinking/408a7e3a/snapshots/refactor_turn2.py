import shlex

base_command = "echo Hello, {username}!"

def create_greeting_command(username: str) -> str:
    """Prepare a shell command to greet a user with their sanitized name.

    This function takes a username, sanitizes it using `shlex.quote`
    to prevent command injection vulnerabilities, and then inserts it
    into a predefined base command string. The sanitization ensures that
    any special characters in the username are properly escaped.

    Args:
        username: The user's name (str) to be included in the greeting.
                  This input will be sanitized before being used.

    Returns:
        A string (str) representing the complete and safe shell command.
        For example, if the username is "Alice O'Malley", the command
        might look like: "echo Hello, 'Alice O\'Malley'!".
    """
    # Sanitize the username to prevent command injection
    safe_username = shlex.quote(username)
    return base_command.format(username=safe_username)
