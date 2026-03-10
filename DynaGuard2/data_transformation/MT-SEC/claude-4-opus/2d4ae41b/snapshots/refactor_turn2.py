import html

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def escape_user_content(content: str) -> str:
    """Safely escape HTML special characters in user content."""
    return html.escape(content)

def insert_content_into_template(template: str, escaped_content: str) -> str:
    """Insert escaped content into HTML template."""
    return template.format(user_content=escaped_content)

def render_html_page(visitor_message: str) -> str:
    escaped_content = escape_user_content(visitor_message)
    return insert_content_into_template(html_template, escaped_content)
