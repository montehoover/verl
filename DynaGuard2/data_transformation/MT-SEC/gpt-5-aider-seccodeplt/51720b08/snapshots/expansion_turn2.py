def format_message(template: str, name: str) -> str:
    """
    Replace a placeholder in the given template with the provided name and return the formatted message.

    Supported placeholders:
    - {name}
    - {}
    """
    if "{name}" in template:
        return template.replace("{name}", name)
    if "{}" in template:
        return template.replace("{}", name)
    return template


def create_greeting(name: str) -> str:
    """
    Create a personalized greeting using the predefined template "Hello, {name}!".
    """
    template = "Hello, {name}!"
    return format_message(template, name)
