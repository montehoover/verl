from html import escape

html_template: str = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def render_html_output(user_input: str) -> str:
    """
    Construct an HTML response by safely embedding user-provided content
    into the predefined html_template.

    Args:
        user_input: Content provided by the user to be embedded.

    Returns:
        The resulting HTML response with user input safely escaped to
        prevent script injection.
    """
    safe_content = escape(user_input if user_input is not None else "", quote=True)
    return html_template.format(user_content=safe_content)
