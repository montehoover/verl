import shlex

base_command = "echo Hello, {recipient}!"

def compose_hello_command(recipient: str) -> str:
    """
    Constructs a shell command to greet a user by inserting their name
    into a predefined template.

    Args:
        recipient: The name of the person to greet.

    Returns:
        A shell command string with the user's name safely included.
    """
    safe_recipient = shlex.quote(recipient)
    return base_command.format(recipient=safe_recipient)
