import shlex

base_command = "echo Hello, {username}!"

def create_greeting_command(username: str) -> str:
    """
    Prepares a shell command to greet a user by inserting their provided name
    into a predefined command string.
    
    Args:
        username: str - The input representing the user's name that will be 
                       included in the generated shell command.
    
    Returns:
        A string representing the final shell command that safely incorporates
        the provided user name.
    """
    # Sanitize the username to prevent command injection
    # shlex.quote() properly escapes the string for safe use in shell commands
    safe_username = shlex.quote(username)
    
    # Replace the placeholder with the sanitized username
    # Note: We use format() here but the username is already safely quoted
    command = base_command.format(username=safe_username)
    
    return command
