html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def escape_html_content(content: str) -> str:
    """
    Escape HTML special characters in the content to prevent injection.
    
    Args:
        content: The raw content to escape
        
    Returns:
        The escaped content safe for HTML insertion
    """
    import html
    return html.escape(content)

def insert_content_into_template(template: str, escaped_content: str) -> str:
    """
    Insert escaped content into the HTML template.
    
    Args:
        template: The HTML template with placeholder
        escaped_content: The pre-escaped content to insert
        
    Returns:
        The complete HTML string with content inserted
    """
    return template.format(user_content=escaped_content)

def build_html_output(client_message: str) -> str:
    """
    Create an HTML response by inserting user content into a given HTML structure.
    
    Args:
        client_message: The content contributed by a user which will be inserted into the HTML template
        
    Returns:
        The resulting HTML string that includes the user's content safely embedded into it
    """
    escaped_content = escape_html_content(client_message)
    return insert_content_into_template(html_template, escaped_content)
