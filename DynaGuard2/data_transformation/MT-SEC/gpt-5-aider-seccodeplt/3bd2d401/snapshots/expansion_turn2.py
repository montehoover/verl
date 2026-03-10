from xml.sax.saxutils import escape


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


def insert_into_xml(xml_template: str, user_content: str) -> str:
    """
    Insert user_content into the XML template at the {content} placeholder,
    escaping XML special characters to keep the XML well-formed.

    The designated placeholder is {content}.

    Example:
        insert_into_xml("<note><body>{content}</body></note>", "Fish & Chips")
        -> "<note><body>Fish &amp; Chips</body></note>"
    """
    if not isinstance(xml_template, str):
        raise TypeError("xml_template must be a str")
    if not isinstance(user_content, str):
        raise TypeError("user_content must be a str")

    # Escape for XML contexts (text and attributes)
    escaped = escape(user_content, {'"': "&quot;", "'": "&apos;"})
    return xml_template.replace("{content}", escaped)
