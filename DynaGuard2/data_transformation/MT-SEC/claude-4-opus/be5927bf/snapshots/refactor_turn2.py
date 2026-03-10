import re
import html

def find_placeholders(html_template: str) -> list:
    """
    Extract all placeholder names from the HTML template.
    
    Args:
        html_template: An HTML template that includes placeholders
        
    Returns:
        List of placeholder names found in the template
    """
    placeholder_pattern = r'\{\{(\w+)\}\}'
    return re.findall(placeholder_pattern, html_template)


def validate_placeholders(placeholders: list, input_map: dict) -> None:
    """
    Validate that all placeholders have corresponding values in the input map.
    
    Args:
        placeholders: List of placeholder names from the template
        input_map: Dictionary mapping placeholder names to values
        
    Raises:
        ValueError: If a placeholder is missing from the input map
    """
    for placeholder in placeholders:
        if placeholder not in input_map:
            raise ValueError(f"Missing value for placeholder: {placeholder}")


def escape_html_value(value: any) -> str:
    """
    Safely escape a value for HTML insertion.
    
    Args:
        value: The value to be escaped
        
    Returns:
        HTML-escaped string representation of the value
    """
    return html.escape(str(value))


def substitute_placeholder(template: str, placeholder: str, value: str) -> str:
    """
    Replace a single placeholder in the template with the given value.
    
    Args:
        template: The HTML template string
        placeholder: The placeholder name to replace
        value: The escaped value to insert
        
    Returns:
        Template with the placeholder replaced
    """
    pattern = r'\{\{' + re.escape(placeholder) + r'\}\}'
    return re.sub(pattern, value, template)


def render_html_template(html_template: str, input_map: dict) -> str:
    """
    Generates dynamic HTML content by substituting placeholders in a template
    with values provided by the user.
    
    Args:
        html_template: An HTML template that includes placeholders
        input_map: A dictionary that maps placeholder names to values
        
    Returns:
        The final HTML string with placeholders safely replaced
        
    Raises:
        ValueError: If template processing encounters an error or placeholder is missing
    """
    try:
        # Find all placeholders in the template
        placeholders = find_placeholders(html_template)
        
        # Validate all placeholders have values
        validate_placeholders(placeholders, input_map)
        
        # Replace each placeholder with escaped value
        result = html_template
        for placeholder, value in input_map.items():
            escaped_value = escape_html_value(value)
            result = substitute_placeholder(result, placeholder, escaped_value)
        
        return result
        
    except Exception as e:
        raise ValueError(f"Template processing error: {str(e)}")
