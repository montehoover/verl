import re
import html

def render_html_template(html_template, input_map):
    """
    Generate dynamic HTML content by replacing placeholders with values.
    
    Args:
        html_template: String containing HTML with placeholders in {placeholder} format
        input_map: Dictionary mapping placeholder names to their values
        
    Returns:
        String with HTML content where placeholders are replaced with escaped values
        
    Raises:
        ValueError: If template processing encounters an error or placeholder is missing
    """
    try:
        # Find all placeholders in the template
        placeholders = re.findall(r'\{([^}]+)\}', html_template)
        
        # Check if all placeholders have corresponding values
        for placeholder in placeholders:
            if placeholder not in input_map:
                raise ValueError(f"Missing value for placeholder: {placeholder}")
        
        # Replace all placeholders with their escaped values
        result = html_template
        for placeholder, value in input_map.items():
            # Escape the value to prevent XSS attacks
            escaped_value = html.escape(str(value))
            result = result.replace(f'{{{placeholder}}}', escaped_value)
        
        return result
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Template processing error: {str(e)}")
