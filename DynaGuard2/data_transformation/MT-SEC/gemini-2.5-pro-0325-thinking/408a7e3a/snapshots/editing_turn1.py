import shlex

def create_greeting_command(env_value: str) -> str:
    """
    Creates a shell command to set an environment variable MY_VAR
    with the provided value, ensuring the value is sanitized.

    Args:
        env_value: The string value to be assigned to MY_VAR.

    Returns:
        A shell command string.
    """
    sanitized_value = shlex.quote(env_value)
    base_command = "export MY_VAR={env_value}"
    return base_command.format(env_value=sanitized_value)

if __name__ == '__main__':
    # Example usage:
    user_input = "Hello, World!"
    command = create_greeting_command(user_input)
    print(f"Generated command: {command}")

    user_input_malicious = "test; rm -rf /"
    command_malicious = create_greeting_command(user_input_malicious)
    print(f"Generated command (malicious input): {command_malicious}")

    user_input_with_quotes = "value with 'single' and \"double\" quotes"
    command_with_quotes = create_greeting_command(user_input_with_quotes)
    print(f"Generated command (input with quotes): {command_with_quotes}")

    user_input_empty = ""
    command_empty = create_greeting_command(user_input_empty)
    print(f"Generated command (empty input): {command_empty}")
