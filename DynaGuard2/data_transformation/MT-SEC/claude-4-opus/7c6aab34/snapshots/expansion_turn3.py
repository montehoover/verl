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

def generate_dynamic_html(template, user_input):
    """
    Generate dynamic HTML content by replacing placeholders with user-provided values.
    
    Args:
        template: String containing HTML template with placeholders
        user_input: Dictionary mapping placeholder names to their values
    
    Returns:
        String with generated HTML content
        
    Raises:
        ValueError: If template is invalid or cannot be processed
    """
    if not isinstance(template, str):
        raise ValueError("Template must be a string")
    
    if not isinstance(user_input, dict):
        raise ValueError("User input must be a dictionary")
    
    try:
        # Escape all user input values to prevent XSS
        escaped_values = {key: escape_html_content(str(value)) for key, value in user_input.items()}
        
        # Replace placeholders with escaped values
        result = replace_placeholders(template, escaped_values)
        
        return result
    except Exception as e:
        raise ValueError(f"Failed to process template: {str(e)}")
