import html

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"


def _escape_html_content(content: str) -> str:
    """Escapes HTML special characters in a string."""
    return html.escape(content)


def _construct_html_page(user_content: str) -> str:
    """Constructs the HTML page with the given user content."""
    return html_template.format(user_content=user_content)


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
    escaped_user_input = _escape_html_content(user_input)
    return _construct_html_page(escaped_user_input)
