import re

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
