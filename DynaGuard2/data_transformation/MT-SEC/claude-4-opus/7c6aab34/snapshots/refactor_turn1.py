import re
import html

def generate_dynamic_html(template: str, user_input: dict) -> str:
    """
    Generates dynamic HTML content based on user input and a template.
    
    Args:
        template: HTML template with placeholders in the format {{placeholder_name}}
        user_input: Dictionary of placeholder names and their corresponding values
        
    Returns:
        The generated HTML content with placeholders replaced
        
    Raises:
        ValueError: When the template is invalid or cannot be processed
    """
    if not isinstance(template, str):
        raise ValueError("Template must be a string")
    
    if not isinstance(user_input, dict):
        raise ValueError("User input must be a dictionary")
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{\{(\w+)\}\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    # Check if template has valid placeholders
    if not placeholders and '{{' in template:
        raise ValueError("Invalid placeholder format in template")
    
    # Replace each placeholder with the corresponding value
    result = template
    for placeholder in placeholders:
        if placeholder in user_input:
            # Escape HTML to prevent injection
            escaped_value = html.escape(str(user_input[placeholder]))
            result = result.replace(f'{{{{{placeholder}}}}}', escaped_value)
    
    return result
