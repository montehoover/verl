import re
import html

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
        # Find all placeholders in the template (e.g., {{placeholder_name}})
        placeholder_pattern = r'\{\{(\w+)\}\}'
        placeholders = re.findall(placeholder_pattern, html_template)
        
        # Check if all placeholders have corresponding values in input_map
        for placeholder in placeholders:
            if placeholder not in input_map:
                raise ValueError(f"Missing value for placeholder: {placeholder}")
        
        # Replace each placeholder with escaped value
        result = html_template
        for placeholder, value in input_map.items():
            # Escape the value to prevent XSS attacks
            escaped_value = html.escape(str(value))
            # Replace the placeholder pattern with the escaped value
            pattern = r'\{\{' + re.escape(placeholder) + r'\}\}'
            result = re.sub(pattern, escaped_value, result)
        
        return result
        
    except Exception as e:
        raise ValueError(f"Template processing error: {str(e)}")
