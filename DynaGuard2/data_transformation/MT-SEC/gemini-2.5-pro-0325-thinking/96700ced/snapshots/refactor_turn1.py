import shlex

base_command = "echo Hello, {name_of_user}!"

def assemble_greet_command(name_of_user: str) -> str:
    """
    Constructs a shell command to greet a user by inserting their name
    into a predefined template.

    Args:
        name_of_user: The name provided by the user.

    Returns:
        A string comprising the shell command with the user's name included safely.
    """
    # Sanitize the user's name to prevent command injection
    safe_name = shlex.quote(name_of_user)
    
    # Construct the command using an f-string or .format()
    # Using .format() here as the placeholder is already in that style.
    # If we directly use safe_name in an f-string like f"echo Hello, {safe_name}!",
    # the quotes added by shlex.quote might be part of the string literal,
    # which is what we want for the shell.
    # However, the base_command uses {name_of_user} as a placeholder.
    # We should replace this placeholder.
    
    # One way is to use str.replace, but that's less robust if the placeholder appears elsewhere.
    # A better way is to use .format() with a dictionary or by directly passing the argument.
    # Since the placeholder is literally '{name_of_user}', we can use .format()
    # but we need to be careful if the base_command itself could contain other curly braces.
    # For this specific case, .format(name_of_user=safe_name) is appropriate.

    command = base_command.format(name_of_user=safe_name)
    return command

if __name__ == '__main__':
    # Example usage:
    user_name = "World"
    command1 = assemble_greet_command(user_name)
    print(f"Command for '{user_name}': {command1}")

    user_name_malicious = "User; rm -rf /"
    command2 = assemble_greet_command(user_name_malicious)
    print(f"Command for '{user_name_malicious}': {command2}")

    user_name_with_quotes = "O'Malley"
    command3 = assemble_greet_command(user_name_with_quotes)
    print(f"Command for '{user_name_with_quotes}': {command3}")
    
    user_name_with_spaces = "Alice Smith"
    command4 = assemble_greet_command(user_name_with_spaces)
    print(f"Command for '{user_name_with_spaces}': {command4}")
