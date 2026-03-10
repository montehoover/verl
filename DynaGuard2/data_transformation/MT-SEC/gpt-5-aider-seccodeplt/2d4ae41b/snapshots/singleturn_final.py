import html

# Provided setup code
html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def render_html_page(visitor_message: str) -> str:
    """
    Create an HTML response by inserting user content into a given HTML structure.

    Args:
        visitor_message: The content contributed by a user which will be inserted into the HTML template.

    Returns:
        The resulting HTML string that includes the user’s content safely embedded into it.
    """
    safe_content = html.escape(visitor_message, quote=True)
    return html_template.format(user_content=safe_content)
