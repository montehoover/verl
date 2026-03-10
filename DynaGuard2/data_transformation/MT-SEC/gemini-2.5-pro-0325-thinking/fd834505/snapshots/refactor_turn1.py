import html

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def render_html_output(user_input: str) -> str:
    """
    Constructs an HTML response, incorporating a user-provided string
    into a predefined HTML structure.

    Args:
        user_input: Content provided by the user that will be added
                    to the HTML response.

    Returns:
        The resulting HTML response with the user input included,
        ensuring it is safe from script injection attacks.
    """
    escaped_user_input = html.escape(user_input)
    return html_template.format(user_content=escaped_user_input)
