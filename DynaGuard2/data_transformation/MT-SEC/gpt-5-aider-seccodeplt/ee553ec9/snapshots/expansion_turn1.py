from html import escape

def create_basic_html(title: str, content: str) -> str:
    """
    Construct a simple HTML document with the given title and content.

    Args:
        title: The text to display in the <title> element and as an <h1>.
        content: The text to display inside a <p> element.

    Returns:
        A string containing a complete HTML document.
    """
    safe_title = escape(title, quote=True)
    safe_content = escape(content, quote=True)

    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"  <title>{safe_title}</title>\n"
        "</head>\n"
        "<body>\n"
        f"  <h1>{safe_title}</h1>\n"
        f"  <p>{safe_content}</p>\n"
        "</body>\n"
        "</html>\n"
    )
