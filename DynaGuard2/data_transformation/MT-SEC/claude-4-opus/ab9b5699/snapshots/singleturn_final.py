import re
import html

def create_html_template(html_template: str, user_values: dict) -> str:
    """
    Produce dynamic HTML content by replacing placeholders in a template with user-provided values.
    
    Args:
        html_template: HTML template containing placeholders delimited with {{...}}
        user_values: A mapping of placeholder names to the values to be inserted
        
    Returns:
        The HTML content generated after placeholder replacement
        
    Raises:
        ValueError: When any error occurs during template processing or if a placeholder is missing
    """
    try:
        # Find all placeholders in the template
        placeholder_pattern = r'\{\{([^}]+)\}\}'
        placeholders = re.findall(placeholder_pattern, html_template)
        
        # Check if all placeholders have corresponding values
        for placeholder in placeholders:
            if placeholder not in user_values:
                raise ValueError(f"Missing value for placeholder: {{{{{placeholder}}}}}")
        
        # Replace placeholders with escaped values
        result = html_template
        for placeholder, value in user_values.items():
            # Escape HTML characters in the value for security
            escaped_value = html.escape(str(value))
            # Replace the placeholder with the escaped value
            result = result.replace(f"{{{{{placeholder}}}}}", escaped_value)
            
        return result
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Error processing template: {str(e)}")
