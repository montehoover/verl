import re
import html

def replace_placeholders(text, values):
    """
    Replace placeholders in a string with values from a dictionary.
    
    Args:
        text: String containing placeholders in the format {placeholder_name}
        values: Dictionary mapping placeholder names to their replacement values
    
    Returns:
        String with placeholders replaced by their corresponding values
    """
    def replacer(match):
        key = match.group(1)
        return str(values.get(key, match.group(0)))
    
    pattern = r'\{([^}]+)\}'
    return re.sub(pattern, replacer, text)

def escape_html_content(text):
    """
    Escape HTML special characters in a string.
    
    Args:
        text: String containing HTML content
    
    Returns:
        String with HTML special characters escaped
    """
    return html.escape(text)
