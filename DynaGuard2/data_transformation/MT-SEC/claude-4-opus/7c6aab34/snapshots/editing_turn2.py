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
    
    # Escape SQL special characters in values
    def escape_sql_value(value):
        # Convert to string first
        str_value = str(value)
        # Escape single quotes by doubling them
        str_value = str_value.replace("'", "''")
        # Escape backslashes
        str_value = str_value.replace("\\", "\\\\")
        # Escape percent signs and underscores for LIKE patterns
        str_value = str_value.replace("%", "\\%")
        str_value = str_value.replace("_", "\\_")
        return str_value
    
    # Replace placeholders with escaped values
    result = template
    for key, value in values.items():
        escaped_value = escape_sql_value(value)
        result = result.replace(f'{{{key}}}', escaped_value)
    
    return result
