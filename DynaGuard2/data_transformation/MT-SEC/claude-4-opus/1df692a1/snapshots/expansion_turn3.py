import re
import html


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
    def replace_match(match):
        key = match.group(1)
        return str(replacements.get(key, match.group(0)))
    
    return re.sub(r'\{\{(\w+)\}\}', replace_match, text)


def create_html_from_template(template, data_mapping):
    """
    Render dynamic HTML content from a template with placeholders.
    
    Args:
        template: String with HTML including {{name}} style placeholders
        data_mapping: Dictionary mapping placeholder keys to values
    
    Returns:
        Processed HTML string with placeholders replaced
    
    Raises:
        ValueError: If a placeholder in the template is not found in data_mapping
    """
    # Find all placeholders in the template
    placeholders = re.findall(r'\{\{(\w+)\}\}', template)
    
    # Check if all placeholders have corresponding values in data_mapping
    missing_keys = [key for key in placeholders if key not in data_mapping]
    if missing_keys:
        raise ValueError(f"Missing keys in data_mapping: {', '.join(missing_keys)}")
    
    # Escape HTML in values and perform replacements
    escaped_mapping = {}
    for key, value in data_mapping.items():
        escaped_mapping[key] = html.escape(str(value))
    
    def replace_match(match):
        key = match.group(1)
        return escaped_mapping.get(key, match.group(0))
    
    return re.sub(r'\{\{(\w+)\}\}', replace_match, template)
