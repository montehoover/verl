import shlex

"""
This module provides a function to safely construct shell commands
for greeting users. It demonstrates secure handling of user input
when incorporating it into command strings.
"""

# Predefined template for the shell command.
# The {name_of_user} placeholder will be replaced by the sanitized user's name.
base_command = "echo Hello, {name_of_user}!"


def assemble_greet_command(name_of_user: str) -> str:
    """
    Constructs a shell command to greet a user by safely inserting their name
    into a predefined command template.

    This function prioritizes security by using `shlex.quote()` to sanitize
    the user-provided name. This prevents command injection
    vulnerabilities by ensuring that the name is treated as a single,
    literal string argument within the shell command, regardless of any
    special characters it may contain. For example, if `name_of_user` is
    "User; rm -rf /", `shlex.quote()` will transform it into something
    like "'User; rm -rf /'", making the entire string a single argument
    to the `echo` command.

    Args:
        name_of_user (str): The name provided by the user. This input is
                            sanitized using `shlex.quote()` before being
                            inserted into the command template.

    Returns:
        str: A string representing the fully constructed shell command,
             with the user's name safely quoted and embedded. For example,
             if `base_command` is "echo Hello, {name_of_user}!" and
             `name_of_user` is "O'Malley", the result could be
             "echo Hello, 'O'\\''Malley'!".

    Raises:
        This function does not define or raise custom exceptions.
        Standard Python exceptions (e.g., `TypeError` from `shlex.quote()`
        if `name_of_user` is not a string, or from `.format()`) may
        occur if inputs are not of the expected type or format.
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
