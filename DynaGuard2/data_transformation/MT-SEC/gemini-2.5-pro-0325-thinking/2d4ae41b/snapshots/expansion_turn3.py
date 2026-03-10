import html

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def embed_user_content(template: str, content: str) -> str:
    """
    Safely embeds user content into an HTML template.

    Args:
        template: The HTML template string with a '{user_content}' placeholder.
        content: The user content string to embed.

    Returns:
        The HTML template with the user content safely embedded.
    """
    escaped_content = html.escape(content)
    return template.replace('{user_content}', escaped_content)

def render_html_page(visitor_message: str) -> str:
    """
    Generates an HTML page with the visitor's message safely embedded.

    Args:
        visitor_message: The message from the visitor.

    Returns:
        A string containing the full HTML page with the message embedded.
    """
    return embed_user_content(html_template, visitor_message)
