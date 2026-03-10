html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def build_html_output(client_message: str) -> str:
    """
    Create an HTML response by inserting user content into a given HTML structure.
    
    Args:
        client_message: The content contributed by a user which will be inserted into the HTML template
        
    Returns:
        The resulting HTML string that includes the user's content safely embedded into it
    """
    import html
    
    # Escape the user content to prevent HTML injection
    escaped_content = html.escape(client_message)
    
    # Insert the escaped content into the template
    return html_template.format(user_content=escaped_content)
