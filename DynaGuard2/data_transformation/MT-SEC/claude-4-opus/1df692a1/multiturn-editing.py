import re
import html

def sanitize_input(user_input: str) -> str:
    """Sanitize user input to prevent HTML/script injection attacks."""
    return html.escape(user_input)

def replace_placeholders(message: str, value_map: dict) -> str:
    """Replace placeholders in a message with values from a dictionary.
    
    Args:
        message: String containing placeholders in the format {placeholder_name}
        value_map: Dictionary mapping placeholder names to replacement values
        
    Returns:
        String with all placeholders replaced
        
    Raises:
        ValueError: If a placeholder in the message doesn't have a corresponding entry in value_map
    """
    # Find all placeholders in the format {placeholder_name}
    placeholders = re.findall(r'\{([^}]+)\}', message)
    
    # Check if all placeholders have corresponding values
    for placeholder in placeholders:
        if placeholder not in value_map:
            raise ValueError(f"Placeholder '{placeholder}' not found in value_map")
    
    # Replace all placeholders
    result = message
    for placeholder, value in value_map.items():
        result = result.replace(f'{{{placeholder}}}', str(value))
    
    return result

def create_html_from_template(template: str, data_mapping: dict) -> str:
    """Build dynamic HTML by substituting placeholders in an HTML template with values.
    
    Args:
        template: HTML structure with placeholders in the format {placeholder_name}
        data_mapping: Dictionary mapping placeholder names to their values
        
    Returns:
        Processed HTML content as a string with all placeholders replaced
        
    Raises:
        ValueError: If there are any processing issues or missing placeholder mappings
    """
    # Find all placeholders in the format {placeholder_name}
    placeholders = re.findall(r'\{([^}]+)\}', template)
    
    # Check if all placeholders have corresponding values
    for placeholder in placeholders:
        if placeholder not in data_mapping:
            raise ValueError(f"Placeholder '{placeholder}' not found in data_mapping")
    
    # Replace all placeholders with sanitized values
    result = template
    for placeholder, value in data_mapping.items():
        # Sanitize the value to prevent HTML/script injection
        sanitized_value = html.escape(str(value))
        result = result.replace(f'{{{placeholder}}}', sanitized_value)
    
    return result
