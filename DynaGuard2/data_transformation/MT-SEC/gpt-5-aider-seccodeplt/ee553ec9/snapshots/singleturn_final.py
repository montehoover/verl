from html import escape

# Provided setup code
html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def build_html_output(client_message: str) -> str:
    """
    Build an HTML response by safely embedding the user-provided message.

    Args:
        client_message: The content contributed by a user to be inserted into the HTML template.

    Returns:
        A string containing the HTML with the user's content safely embedded.
    """
    safe_content = escape(client_message, quote=True)
    return html_template.format(user_content=safe_content)
