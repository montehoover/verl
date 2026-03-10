import shlex

base_command = "echo Hello, {recipient}!"

def format_message(template: str, name: str) -> str:
    """
    Replaces a placeholder in the template with the provided name.

    Args:
        template: The string template with a placeholder for the name.
                  It is assumed the placeholder is "{name}".
        name: The name to insert into the template.

    Returns:
        The formatted message string.
    """
    return template.replace("{name}", name)


def create_greeting(name: str) -> str:
    """
    Creates a personalized greeting message.

    Args:
        name: The name to include in the greeting.

    Returns:
        A personalized greeting message string.
    """
    greeting_template = "Hello, {name}!"
    return format_message(greeting_template, name)


def compose_hello_command(recipient: str) -> str:
    """
    Constructs a shell command to greet a user by name safely.

    Args:
        recipient: The name of the user to greet.

    Returns:
        A shell command string.
    """
    # Use shlex.quote to ensure the recipient's name is safely included in the command
    safe_recipient = shlex.quote(recipient)
    return base_command.replace("{recipient}", safe_recipient)
