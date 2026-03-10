from html import escape

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def sanitize_user_content(client_message: str) -> str:
    """
    Return an HTML-escaped representation of user-provided content.
    """
    return escape(client_message, quote=True)

def insert_user_content_into_template(template: str, user_content: str) -> str:
    """
    Insert sanitized user content into the provided HTML template.
    """
    return template.format(user_content=user_content)

def build_html_output(client_message: str) -> str:
    """
    Build an HTML response by inserting user content into a predefined template.

    Args:
        client_message: The content contributed by a user which will be inserted into the HTML template.

    Returns:
        The resulting HTML string that includes the user’s content safely embedded into it.
    """
    safe_user_content = sanitize_user_content(client_message)
    return insert_user_content_into_template(html_template, safe_user_content)
