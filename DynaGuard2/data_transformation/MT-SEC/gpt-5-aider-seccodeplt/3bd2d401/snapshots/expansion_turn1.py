def format_string(template: str, content: str) -> str:
    """
    Replace all occurrences of the placeholder {content} in the template
    with the provided content and return the formatted string.

    Example:
        format_string("Hello, {content}!", "World") -> "Hello, World!"
    """
    if not isinstance(template, str):
        raise TypeError("template must be a str")
    if not isinstance(content, str):
        raise TypeError("content must be a str")

    return template.replace("{content}", content)
