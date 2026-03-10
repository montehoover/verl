from html import escape

html_template: str = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def _sanitize_user_input(user_input: str) -> str:
    """
    Pure function that escapes user-provided input to prevent script injection.
    """
    return escape(user_input if user_input is not None else "", quote=True)

def _build_html_response(user_content: str, template: str = html_template) -> str:
    """
    Pure function that inserts the sanitized content into the HTML template.
    """
    return template.format(user_content=user_content)

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
    safe_content = _sanitize_user_input(user_input)
    return _build_html_response(safe_content)
