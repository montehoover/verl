def replace_placeholders(templates, values):
    """
    Replace placeholders in one or more templates with values from a dictionary.
    
    Args:
        templates: String containing placeholders in {placeholder} format, or a list of such strings
        values: Dictionary mapping placeholder names to their values
        
    Returns:
        String with all placeholders replaced if single template provided,
        or list of strings if multiple templates provided
        
    Raises:
        ValueError: If a placeholder is not found in the values dictionary
    """
    import re
    
    # Handle single template as a list for uniform processing
    if isinstance(templates, str):
        templates_list = [templates]
        single_template = True
    else:
        templates_list = templates
        single_template = False
    
    results = []
    
    for template in templates_list:
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
        
        results.append(result)
    
    # Return single string if input was single template, otherwise return list
    return results[0] if single_template else results
