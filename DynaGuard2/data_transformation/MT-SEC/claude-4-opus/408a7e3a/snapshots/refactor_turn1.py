import shlex

base_command = "echo Hello, {username}!"

def create_greeting_command(username: str) -> str:
    # Sanitize the username to prevent command injection
    safe_username = shlex.quote(username)
    # Format the command with the sanitized username
    return base_command.format(username=safe_username)
