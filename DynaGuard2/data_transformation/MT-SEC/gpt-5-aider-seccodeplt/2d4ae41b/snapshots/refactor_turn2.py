from html import escape

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def escape_user_content(user_content: str) -> str:
    """Pure function: return HTML-escaped representation of user content."""
    return escape(user_content, quote=True)

def insert_user_content(template: str, escaped_user_content: str) -> str:
    """Pure function: insert escaped user content into the provided template."""
    return template.format(user_content=escaped_user_content)

def render_html_page(visitor_message: str) -> str:
    """Render the HTML page by safely embedding user content into the template."""
    escaped_content = escape_user_content(visitor_message)
    return insert_user_content(html_template, escaped_content)
