html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def build_html_output(client_message: str) -> str:
    """
    Create an HTML response by inserting user content into a given HTML structure.
    
    Args:
        client_message: The content contributed by a user which will be inserted into the HTML template.
    
    Returns:
        The resulting HTML string that includes the user's content safely embedded into it.
    """
    return html_template.format(user_content=client_message)
