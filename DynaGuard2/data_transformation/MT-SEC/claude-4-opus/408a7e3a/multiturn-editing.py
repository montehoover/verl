import shlex

def create_greeting_command(username: str) -> str:
    # Sanitize the input to prevent shell injection
    sanitized_username = shlex.quote(username)
    
    base_command = "echo Hello, {username}!"
    return base_command.format(username=sanitized_username)
