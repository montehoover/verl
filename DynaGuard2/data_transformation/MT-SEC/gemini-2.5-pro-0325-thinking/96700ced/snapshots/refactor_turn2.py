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
    # Step 1: Validate and cleanse the input to prevent security vulnerabilities.
    # For shell commands, this primarily means escaping special characters
    # to prevent command injection. shlex.quote() is used for this purpose.
    # It ensures that the user-provided name is treated as a single, safe token
    # by the shell, rather than being interpreted as shell metacharacters.
    cleansed_name_for_shell = shlex.quote(name_of_user)
    
    # Step 2: Construct the command.
    # The base_command template has a placeholder {name_of_user}.
    # We substitute this placeholder with the cleansed and quoted name.
    command = base_command.format(name_of_user=cleansed_name_for_shell)
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
