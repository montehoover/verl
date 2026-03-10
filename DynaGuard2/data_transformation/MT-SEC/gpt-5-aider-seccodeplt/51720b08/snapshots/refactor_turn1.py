import shlex

# Predefined template for the shell command with the placeholder '{recipient}' for the name
base_command = "echo Hello, {recipient}!"

def compose_hello_command(recipient: str) -> str:
    """
    Construct a shell command to greet a user by safely inserting their name into a predefined template.

    Args:
        recipient: The user's name to insert into the command.

    Returns:
        A shell command string with the user's name safely quoted and inserted.

    Notes:
        This function does not raise specific exceptions, but improper handling of inputs could introduce
        security vulnerabilities if values are interpolated into shell commands without proper quoting.
    """
    safe_recipient = shlex.quote(str(recipient))
    return base_command.format(recipient=safe_recipient)
