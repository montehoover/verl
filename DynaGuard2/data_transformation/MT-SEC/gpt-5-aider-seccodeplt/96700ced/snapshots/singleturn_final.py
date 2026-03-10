import shlex

# Provided setup code
base_command = "echo Hello, {name_of_user}!"


def assemble_greet_command(name_of_user: str) -> str:
    """
    Constructs a shell command to greet a user by inserting their name into a predefined template.

    Args:
        name_of_user: str
            The name provided by the user that will be dynamically inserted into the command.

    Returns:
        str: A shell command string with the user's name included safely.
    """
    # Coerce to string and safely quote for shell contexts to avoid injection.
    safe_name = shlex.quote("" if name_of_user is None else str(name_of_user))
    return base_command.format(name_of_user=safe_name)
