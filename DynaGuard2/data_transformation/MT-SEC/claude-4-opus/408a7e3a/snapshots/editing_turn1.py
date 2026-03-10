import shlex

def create_greeting_command(env_value: str) -> str:
    # Sanitize the input to prevent shell injection
    sanitized_value = shlex.quote(env_value)
    
    base_command = "export MY_VAR={env_value}"
    return base_command.format(env_value=sanitized_value)
