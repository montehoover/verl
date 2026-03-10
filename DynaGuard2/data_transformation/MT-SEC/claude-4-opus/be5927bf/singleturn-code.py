import re
import html

def render_html_template(html_template: str, input_map: dict) -> str:
    """
    Generates dynamic HTML content by substituting placeholders in a template with values provided by the user.
    
    Args:
        html_template: str, an HTML template that includes placeholders in {{placeholder}} format
        input_map: dict, a dictionary that maps placeholder names to the values to be inserted
    
    Returns:
        str: The final HTML string with the placeholders safely replaced with appropriate values
    
    Raises:
        ValueError: Raised if the template processing encounters an error or if a placeholder is missing from the input
    """
    try:
        # Find all placeholders in the template using regex
        # Pattern matches {{placeholder_name}} format
        placeholder_pattern = r'\{\{(\w+)\}\}'
        placeholders = re.findall(placeholder_pattern, html_template)
        
        # Check if all placeholders have corresponding values in input_map
        for placeholder in placeholders:
            if placeholder not in input_map:
                raise ValueError(f"Missing value for placeholder: {placeholder}")
        
        # Replace each placeholder with its escaped value
        result = html_template
        for placeholder in placeholders:
            # Get the value from input_map
            value = input_map[placeholder]
            
            # Convert value to string and escape HTML special characters
            escaped_value = html.escape(str(value))
            
            # Replace the placeholder with the escaped value
            placeholder_with_braces = f"{{{{{placeholder}}}}}"
            result = result.replace(placeholder_with_braces, escaped_value)
        
        return result
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Error processing template: {str(e)}")
