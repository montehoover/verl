import re
import html

def build_html_content(html_template, values_dict):
    """
    Replace placeholders in HTML template with values from values_dict.
    
    Args:
        html_template: String containing HTML with placeholders in {{placeholder}} format
        values_dict: Dictionary mapping placeholder names to replacement values
        
    Returns:
        String with all placeholders replaced with HTML-escaped values
        
    Raises:
        ValueError: If a placeholder in template doesn't exist in values_dict
    """
    # Find all placeholders in the template
    placeholders = re.findall(r'\{\{([^}]+)\}\}', html_template)
    
    # Check if all placeholders have corresponding values
    for placeholder in placeholders:
        if placeholder not in values_dict:
            raise ValueError(f"No replacement found for placeholder: {{{{{placeholder}}}}}")
    
    # Replace all placeholders with their HTML-escaped values
    result = html_template
    for placeholder, value in values_dict.items():
        escaped_value = html.escape(str(value))
        result = result.replace(f"{{{{{placeholder}}}}}", escaped_value)
    
    return result
