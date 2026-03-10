import re
import html

def generate_dynamic_html(template, user_input):
    """
    Generate HTML content by replacing placeholders in a template with user-provided values.
    
    Args:
        template: String with HTML content and placeholders in {key} format
        user_input: Dictionary with keys corresponding to placeholders
        
    Returns:
        Generated HTML content with placeholders replaced
        
    Raises:
        ValueError: If template is invalid or cannot be processed
    """
    # Check if template is a valid string
    if not isinstance(template, str):
        raise ValueError("Template must be a string")
    
    # Check for malformed placeholders (unclosed braces)
    if template.count('{') != template.count('}'):
        raise ValueError("Invalid template: mismatched braces")
    
    # Find all placeholders in the template
    placeholders = re.findall(r'\{([^}]+)\}', template)
    
    # Check that all placeholders have corresponding values
    for placeholder in placeholders:
        if placeholder not in user_input:
            raise ValueError(f"Missing value for placeholder: {placeholder}")
    
    # Replace placeholders with HTML-escaped values
    result = template
    for key, value in user_input.items():
        # HTML escape the value to prevent XSS
        escaped_value = html.escape(str(value))
        result = result.replace(f'{{{key}}}', escaped_value)
    
    return result
