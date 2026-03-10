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


def format_template(text, replacements):
    """
    Replace placeholders in text with values from replacements dictionary.
    Handles {{name}} style placeholders and gracefully ignores missing keys.
    
    Args:
        text: String containing placeholders in {{name}} format
        replacements: Dictionary mapping placeholder names to replacement values
    
    Returns:
        String with placeholders replaced where keys exist
    """
    import re
    
    def replace_match(match):
        key = match.group(1)
        return str(replacements.get(key, match.group(0)))
    
    return re.sub(r'\{\{(\w+)\}\}', replace_match, text)
