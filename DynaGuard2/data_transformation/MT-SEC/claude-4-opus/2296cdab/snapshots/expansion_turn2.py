def sanitize_text(text):
    """
    Sanitizes input text by escaping HTML special characters.
    
    Args:
        text: The input string to sanitize
        
    Returns:
        String with HTML special characters escaped
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Replace HTML special characters with their escaped equivalents
    replacements = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }
    
    # Apply replacements in the correct order (& must be first)
    result = text
    for char, escaped in replacements.items():
        result = result.replace(char, escaped)
    
    return result


def wrap_with_tag(tag, content):
    """
    Wraps content with the specified HTML tag.
    
    Args:
        tag: The HTML tag name (without angle brackets)
        content: The content to wrap
        
    Returns:
        String with content wrapped in HTML tags
    """
    return f"<{tag}>{content}</{tag}>"
