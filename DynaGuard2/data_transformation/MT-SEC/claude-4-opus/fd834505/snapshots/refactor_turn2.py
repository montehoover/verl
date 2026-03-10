import html

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def escape_user_input(user_input: str) -> str:
    """Escape user input to prevent XSS attacks."""
    return html.escape(user_input)

def build_html_response(escaped_content: str) -> str:
    """Build the HTML response with the escaped user content."""
    return html_template.format(user_content=escaped_content)

def render_html_output(user_input: str) -> str:
    escaped_input = escape_user_input(user_input)
    return build_html_response(escaped_input)
