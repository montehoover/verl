import shlex

def create_greeting_command(username: str) -> str:
    """
    Creates a shell command for a friendly greeting using the user's name,
    ensuring the name is sanitized.

    Args:
        username: The user's name to be included in the greeting.

    Returns:
        A shell command string.
    """
    sanitized_username = shlex.quote(username)
    base_command = "echo Hello, {username}!"
    return base_command.format(username=sanitized_username)

if __name__ == '__main__':
    # Example usage:
    user_name_input = "Alice"
    command = create_greeting_command(user_name_input)
    print(f"Generated command: {command}")

    user_name_malicious = "Bob; sudo reboot"
    command_malicious = create_greeting_command(user_name_malicious)
    print(f"Generated command (malicious input): {command_malicious}")

    user_name_with_quotes = "Charlie 'Chuck' Brown"
    command_with_quotes = create_greeting_command(user_name_with_quotes)
    print(f"Generated command (input with quotes): {command_with_quotes}")

    user_name_empty = ""
    command_empty = create_greeting_command(user_name_empty)
    print(f"Generated command (empty input): {command_empty}")
