from html import escape

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def render_html_page(visitor_message: str) -> str:
    escaped_content = escape(visitor_message, quote=True)
    return html_template.format(user_content=escaped_content)
