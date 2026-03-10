import re

def build_html_content(template, replacements):
    """
    Replace placeholders in template with values from replacements dictionary.
    
    Args:
        template: String containing placeholders in {{placeholder}} format (can be JSON)
        replacements: Dictionary mapping placeholder names to replacement values
        
    Returns:
        String with all placeholders replaced
        
    Raises:
        ValueError: If a placeholder in template doesn't exist in replacements
    """
    # Find all placeholders in the template
    placeholders = re.findall(r'\{\{([^}]+)\}\}', template)
    
    # Check if all placeholders have corresponding replacements
    for placeholder in placeholders:
        if placeholder not in replacements:
            raise ValueError(f"No replacement found for placeholder: {{{{{placeholder}}}}}")
    
    # Replace all placeholders with their values
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(f"{{{{{placeholder}}}}}", str(value))
    
    return result
