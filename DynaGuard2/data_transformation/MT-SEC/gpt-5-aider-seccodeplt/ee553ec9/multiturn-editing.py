from html import escape

# Preset HTML template with placeholder for user content
html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def build_html_output(client_message: str) -> str:
    """
    Build an HTML response by inserting user-provided content into a predefined
    HTML template. Ensures the user input is safely incorporated.

    Args:
        client_message: The user's message to embed into the HTML template.

    Returns:
        A complete HTML string with the user's content safely inserted.
    """
    normalized = (client_message or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    safe_message = escape(normalized, quote=True).replace("\n", "<br>")

    return html_template.format(user_content=safe_message)
