from html import escape

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

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

def insert_user_content(html_template: str, user_message: str) -> str:
    """
    Insert the user_message into the {user_content} placeholder of html_template
    using Python's string formatting.

    Args:
        html_template: An HTML template string containing a {user_content} placeholder.
        user_message: The user-supplied message to insert.

    Returns:
        A new HTML string with the user content inserted.
    """
    safe_message = escape(user_message, quote=True)
    return html_template.format(user_content=safe_message)

def build_html_output(client_message: str) -> str:
    """
    Assemble an HTML response by embedding the client_message securely into the html_template.
    """
    return insert_user_content(html_template, client_message)
