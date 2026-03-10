import shlex

base_command = "echo Hello, {name_of_user}!"

def assemble_greet_command(name_of_user: str) -> str:
    """
    Constructs a shell command to greet a user by inserting their name 
    into a predefined template.

    Args:
        name_of_user: str, the name provided by the user that will be 
                      dynamically inserted into the command.

    Returns:
        A string comprising the shell command with the user's name included safely.

    Raises:
        This function does not raise specific exceptions but improper handling 
        of inputs could introduce security vulnerabilities if the output string 
        is directly executed without further sanitization in a shell context 
        where `shlex.quote` is insufficient (though it's generally robust).
    """
    # Sanitize the user input to prevent shell injection
    safe_name_of_user = shlex.quote(name_of_user)
    
    # Construct the command. Note: The placeholder in base_command does not need
    # to be quoted again as shlex.quote handles the entire string.
    # However, if we were building the command piece by piece, each piece
    # coming from user input would need quoting.
    # Here, we are formatting a string template, so we use the quoted name.
    # The base_command itself is trusted.
    command = base_command.format(name_of_user=safe_name_of_user)
    return command

if __name__ == '__main__':
    # Example Usage:
    user_name = "Alice"
    command = assemble_greet_command(user_name)
    print(f"Input name: {user_name}")
    print(f"Generated command: {command}")

    user_name_malicious = "Alice; rm -rf /"
    command_malicious = assemble_greet_command(user_name_malicious)
    print(f"Input name: {user_name_malicious}")
    print(f"Generated command: {command_malicious}")

    user_name_with_spaces = "Bob The Builder"
    command_with_spaces = assemble_greet_command(user_name_with_spaces)
    print(f"Input name: {user_name_with_spaces}")
    print(f"Generated command: {command_with_spaces}")
