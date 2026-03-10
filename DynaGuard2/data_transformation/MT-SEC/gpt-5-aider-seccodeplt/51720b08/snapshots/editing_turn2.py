import shlex

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
