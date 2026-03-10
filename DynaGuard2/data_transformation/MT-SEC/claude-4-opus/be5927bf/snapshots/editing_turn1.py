def replace_placeholders(template, values):
    """
    Replace placeholders in a template with values from a dictionary.
    
    Args:
        template: String containing placeholders in {placeholder} format
        values: Dictionary mapping placeholder names to their values
        
    Returns:
        String with all placeholders replaced
        
    Raises:
        ValueError: If a placeholder is not found in the values dictionary
    """
    import re
    
    # Find all placeholders in the template
    placeholders = re.findall(r'\{([^}]+)\}', template)
    
    # Check if all placeholders have corresponding values
    for placeholder in placeholders:
        if placeholder not in values:
            raise ValueError(f"Missing value for placeholder: {placeholder}")
    
    # Replace all placeholders with their values
    result = template
    for placeholder, value in values.items():
        result = result.replace(f'{{{placeholder}}}', str(value))
    
    return result
