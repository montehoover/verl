import shlex

base_command = "echo Hello, {recipient}!"

def compose_greeting_command(name: str) -> str:
    """
    Compose a command-line string to greet the user.

    Args:
        name: The user's name.

    Returns:
        A command-line instruction like: print 'Hello, [name]!'
        The greeting is safely shell-quoted to handle spaces and special characters.
    """
    message = f"Hello, {name}!"
    return f"print {shlex.quote(message)}"

def compose_hello_command(recipient: str) -> str:
    """
    Compose a shell command using the base_command template that safely includes the recipient's name.

    Args:
        recipient: The user's name to include in the greeting.

    Returns:
        A shell command string like: echo Hello, 'Alice'!
        The recipient is shell-quoted to prevent command injection.
    """
    safe_recipient = shlex.quote(str(recipient))
    return base_command.format(recipient=safe_recipient)
