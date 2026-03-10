from xml.sax.saxutils import escape


def create_greeting(name: str) -> str:
    """
    Return a greeting message of the form 'Hello, {name}!'.

    Args:
        name: The person's name.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"


def create_xml_message(content: str) -> str:
    """
    Return a simple XML-like message of the form '<message>{content}</message>'.

    Args:
        content: The message content to insert.

    Returns:
        The formatted XML-like string.
    """
    return f"<message>{content}</message>"


# Predefined XML template with a placeholder for user input
xml_design: str = "<response><message>{input_data}</message></response>"


def process_xml_reply(input_data: str) -> str:
    """
    Safely insert user-supplied content into the XML template defined by `xml_design`.

    Args:
        input_data: The raw user content to place inside the <message> element.

    Returns:
        The final XML response string with the content safely escaped and inserted.
    """
    try:
        safe_content = escape(input_data, {'"': '&quot;', "'": '&apos;'})
    except Exception:
        # Coerce to string and escape if input_data isn't a string or escaping fails
        try:
            safe_content = escape(str(input_data), {'"': '&quot;', "'": '&apos;'})
        except Exception:
            safe_content = ""

    try:
        return xml_design.replace("{input_data}", safe_content)
    except Exception:
        # Fallback to a minimal valid structure if the template is not usable
        return f"<response><message>{safe_content}</message></response>"
