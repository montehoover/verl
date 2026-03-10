import shlex

# Predefined template for the shell command
base_command = "echo Hello, {name_of_user}!"

def assemble_greet_command(name_of_user: str) -> str:
    """
    Construct a shell command that greets a user, safely inserting their name.

    Args:
        name_of_user (str): The user's name to insert into the command.

    Returns:
        str: The shell command string with the user's name safely included.
    """
    safe_name = shlex.quote(str(name_of_user))
    return base_command.format(name_of_user=safe_name)
