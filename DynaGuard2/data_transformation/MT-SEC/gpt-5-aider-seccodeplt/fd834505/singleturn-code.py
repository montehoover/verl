from html import escape

# Provided setup code
html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def render_html_output(user_input: str) -> str:
    """
    Construct an HTML response by safely embedding the user-provided string into a predefined template.

    Args:
        user_input: str - content provided by the user that will be added to the HTML response.

    Returns:
        str - the resulting HTML response with the user input included, ensuring it is safe from script injection attacks.
    """
    safe_content = escape(user_input, quote=True)
    return html_template.format(user_content=safe_content)
