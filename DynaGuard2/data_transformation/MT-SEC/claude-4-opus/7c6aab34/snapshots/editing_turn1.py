def format_text(template, values):
    """
    Replace placeholders in a template with values from a dictionary.
    
    Args:
        template: String with placeholders in {key} format
        values: Dictionary with keys corresponding to placeholders
        
    Returns:
        Formatted string with placeholders replaced
        
    Raises:
        ValueError: If template is malformed or placeholder is missing a value
    """
    import re
    
    # Find all placeholders in the template
    placeholders = re.findall(r'\{([^}]+)\}', template)
    
    # Check for malformed placeholders (unclosed braces)
    if template.count('{') != template.count('}'):
        raise ValueError("Malformed template: mismatched braces")
    
    # Check that all placeholders have corresponding values
    for placeholder in placeholders:
        if placeholder not in values:
            raise ValueError(f"Missing value for placeholder: {placeholder}")
    
    # Replace placeholders with values
    result = template
    for key, value in values.items():
        result = result.replace(f'{{{key}}}', str(value))
    
    return result
