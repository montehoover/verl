def replace_placeholder(text, replacements):
    """
    Replace placeholders in text with values from replacements dictionary.
    
    Args:
        text: String containing placeholders
        replacements: Dictionary mapping placeholder names to replacement values
    
    Returns:
        String with placeholders replaced
    """
    result = text
    for placeholder, value in replacements.items():
        result = result.replace(f"{{{placeholder}}}", str(value))
    return result
