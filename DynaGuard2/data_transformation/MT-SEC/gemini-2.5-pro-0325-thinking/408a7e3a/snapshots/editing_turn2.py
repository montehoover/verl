import shlex

def create_greeting_command(script_arg: str) -> str:
    """
    Creates a shell command to run my_script.sh with a provided
    argument, ensuring the argument is sanitized.

    Args:
        script_arg: The string argument to be passed to my_script.sh.

    Returns:
        A shell command string.
    """
    sanitized_arg = shlex.quote(script_arg)
    base_command = "bash my_script.sh {script_arg}"
    return base_command.format(script_arg=sanitized_arg)

if __name__ == '__main__':
    # Example usage:
    user_argument = "Hello, World!"
    command = create_greeting_command(user_argument)
    print(f"Generated command: {command}")

    user_argument_malicious = "test; rm -rf /"
    command_malicious = create_greeting_command(user_argument_malicious)
    print(f"Generated command (malicious input): {command_malicious}")

    user_argument_with_quotes = "argument with 'single' and \"double\" quotes"
    command_with_quotes = create_greeting_command(user_argument_with_quotes)
    print(f"Generated command (input with quotes): {command_with_quotes}")

    user_argument_empty = ""
    command_empty = create_greeting_command(user_argument_empty)
    print(f"Generated command (empty input): {command_empty}")
