from html import escape

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def build_html_output(client_message: str) -> str:
    """
    Build an HTML response by inserting user content into a predefined template.

    Args:
        client_message: The content contributed by a user which will be inserted into the HTML template.

    Returns:
        The resulting HTML string that includes the user’s content safely embedded into it.
    """
    safe_user_content = escape(client_message, quote=True)
    return html_template.format(user_content=safe_user_content)
