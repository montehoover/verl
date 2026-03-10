html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def render_html_page(visitor_message: str) -> str:
    """
    Creates an HTML response by inserting user content into a given HTML structure.
    
    Args:
        visitor_message: The content contributed by a user which will be inserted into the HTML template.
    
    Returns:
        The resulting HTML string that includes the user's content embedded into it.
    """
    return html_template.format(user_content=visitor_message)
