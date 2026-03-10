import shlex

# Predefined template for the shell command with the placeholder '{recipient}'
base_command = "echo Hello, {recipient}!"

def compose_hello_command(recipient: str) -> str:
    """
    Construct a shell command to greet a user by inserting their name into a predefined template.

    Args:
        recipient (str): The name provided by the user that will be dynamically inserted.

    Returns:
        str: The shell command with the user's name included safely.

    Notes:
        This function does not raise specific exceptions but improper handling of inputs
        could introduce security vulnerabilities. To mitigate shell injection risks, the
        recipient is safely shell-escaped before insertion.
    """
    safe_recipient = shlex.quote(str(recipient))
    return base_command.format(recipient=safe_recipient)
