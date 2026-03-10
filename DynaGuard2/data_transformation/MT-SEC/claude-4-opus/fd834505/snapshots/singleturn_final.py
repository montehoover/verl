import html

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def render_html_output(user_input: str) -> str:
    """
    Construct an HTML response incorporating user-provided string into a predefined HTML structure.
    
    Args:
        user_input: str, content provided by the user that will be added to the HTML response
        
    Returns:
        str, the resulting HTML response with the user input included, 
        ensuring it is safe from script injection attacks
    """
    # Escape the user input to prevent XSS attacks
    escaped_content = html.escape(user_input)
    
    # Insert the escaped content into the HTML template
    result = html_template.format(user_content=escaped_content)
    
    return result
