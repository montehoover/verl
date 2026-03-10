import shlex

base_command = "export MY_VAR={env_value}"

def create_greeting_command(env_value: str) -> str:
    """
    Create a shell command to set MY_VAR safely using the provided value.
    The value is shell-escaped to prevent shell injection.
    """
    if not isinstance(env_value, str):
        env_value = str(env_value)
    sanitized = shlex.quote(env_value)
    return base_command.format(env_value=sanitized)
