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


def insert_into_xml(xml_template, user_content):
    """
    Insert user content into an XML template at the designated placeholder.
    
    Args:
        xml_template (str): The XML template string containing a placeholder
        user_content (str): The content to insert into the XML template
    
    Returns:
        str: The resulting XML string with the placeholder replaced
    """
    # Replace {user_content} placeholder with the provided user content
    return xml_template.replace("{user_content}", user_content)
