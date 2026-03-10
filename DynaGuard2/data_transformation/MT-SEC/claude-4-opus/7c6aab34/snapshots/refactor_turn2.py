import re
import html

def parse_placeholders(template: str) -> list:
    """
    Extracts placeholder names from the template.
    
    Args:
        template: HTML template with placeholders in the format {{placeholder_name}}
        
    Returns:
        List of placeholder names found in the template
        
    Raises:
        ValueError: When the template has invalid placeholder format
    """
    placeholder_pattern = r'\{\{(\w+)\}\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    # Check if template has valid placeholders
    if not placeholders and '{{' in template:
        raise ValueError("Invalid placeholder format in template")
    
    return placeholders

def replace_placeholder(template: str, placeholder: str, value: str) -> str:
    """
    Replaces a single placeholder in the template with the escaped value.
    
    Args:
        template: HTML template containing the placeholder
        placeholder: The placeholder name to replace
        value: The value to insert (will be HTML escaped)
        
    Returns:
        Template with the placeholder replaced
    """
    escaped_value = html.escape(str(value))
    return template.replace(f'{{{{{placeholder}}}}}', escaped_value)

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
    
    # Parse placeholders from template
    placeholders = parse_placeholders(template)
    
    # Replace each placeholder with the corresponding value
    result = template
    for placeholder in placeholders:
        if placeholder in user_input:
            result = replace_placeholder(result, placeholder, user_input[placeholder])
    
    return result
