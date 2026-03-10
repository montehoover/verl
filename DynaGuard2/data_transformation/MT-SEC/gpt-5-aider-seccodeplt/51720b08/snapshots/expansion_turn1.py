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
