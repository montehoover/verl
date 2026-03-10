import shlex

base_command = "echo Hello, {username}!"

def create_greeting_command(username: str) -> str:
    """
    Create a shell command that echoes a greeting with the provided username.
    The username is shell-escaped to prevent injection.
    """
    if not isinstance(username, str):
        username = str(username)
    sanitized = shlex.quote(username)
    return base_command.format(username=sanitized)
