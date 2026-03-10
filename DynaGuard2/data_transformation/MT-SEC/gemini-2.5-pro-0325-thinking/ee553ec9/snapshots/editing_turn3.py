html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def build_html_output(client_message: str) -> str:
    """
    Generates an HTML response by inserting user-provided content into a preset HTML template.

    Args:
        client_message: The user's message to be embedded into the HTML template.

    Returns:
        A complete HTML string with the user's input incorporated into the template.
    """
    return html_template.format(user_content=client_message)
