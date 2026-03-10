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
    import re
    
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
