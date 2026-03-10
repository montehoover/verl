def format_string(template, content):
    """
    Replace a placeholder in the template with the provided content.
    
    Args:
        template (str): The template string containing a placeholder
        content (str): The content to insert into the template
    
    Returns:
        str: The formatted string with the placeholder replaced
    """
    # Replace {content} placeholder with the provided content
    return template.replace("{content}", content)
