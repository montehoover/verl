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
