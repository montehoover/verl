def build_html_output(client_message: str) -> str:
    """
    Generate an HTML email body by inserting user-provided details into an email template.
    
    Args:
        client_message: The message to personalize the email's content
        
    Returns:
        A complete HTML string with the user's message inside the email body
    """
    html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"
    
    return html_template.format(user_content=client_message)
