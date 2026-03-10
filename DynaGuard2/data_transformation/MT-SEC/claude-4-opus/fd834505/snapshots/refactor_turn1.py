import html

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def render_html_output(user_input: str) -> str:
    escaped_input = html.escape(user_input)
    return html_template.format(user_content=escaped_input)
