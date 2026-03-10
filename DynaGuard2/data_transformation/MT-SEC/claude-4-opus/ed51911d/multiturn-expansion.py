import re
import html

def parse_placeholders(html_string):
    """
    Parse HTML string and extract all placeholder names formatted as {{...}}
    Handles nested placeholders by finding the outermost {{ }} pairs
    
    Args:
        html_string (str): HTML string containing placeholders
        
    Returns:
        list: List of placeholder names found in the HTML string
    """
    placeholders = []
    
    # Pattern to match {{...}} including nested braces
    # This finds the outermost {{ }} pairs
    pattern = r'\{\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}\}'
    
    matches = re.findall(pattern, html_string)
    
    for match in matches:
        # Check if this match contains nested placeholders
        nested_matches = re.findall(r'\{\{([^{}]+)\}\}', match)
        if nested_matches:
            placeholders.extend(nested_matches)
        else:
            placeholders.append(match.strip())
    
    return placeholders


def replace_placeholders(html_template, values, default_value=''):
    """
    Replace placeholders in HTML template with values from dictionary
    
    Args:
        html_template (str): HTML string containing {{placeholder}} patterns
        values (dict): Dictionary mapping placeholder names to replacement values
        default_value (str): Default value for missing placeholders
        
    Returns:
        str: HTML string with placeholders replaced by escaped values
    """
    # Create a copy of the template to modify
    result = html_template
    
    # Find all placeholders using the parse function
    placeholders = parse_placeholders(html_template)
    
    # Process each unique placeholder
    for placeholder in set(placeholders):
        # Get the value from dictionary or use default
        replacement = values.get(placeholder, default_value)
        
        # Escape HTML to prevent injection
        escaped_replacement = html.escape(str(replacement))
        
        # Replace all occurrences of this placeholder
        pattern = r'\{\{' + re.escape(placeholder) + r'\}\}'
        result = re.sub(pattern, escaped_replacement, result)
    
    return result


def build_html_content(html_template, values_dict):
    """
    Process HTML template by replacing placeholders with values from dictionary
    
    Args:
        html_template (str): HTML string containing {{placeholder}} patterns
        values_dict (dict): Dictionary mapping placeholder names to replacement values
        
    Returns:
        str: Final HTML string with all placeholders replaced
        
    Raises:
        ValueError: If a placeholder is missing from values_dict
    """
    # Find all placeholders in the template
    placeholders = parse_placeholders(html_template)
    
    # Check for missing placeholders
    missing_placeholders = []
    for placeholder in set(placeholders):
        if placeholder not in values_dict:
            missing_placeholders.append(placeholder)
    
    if missing_placeholders:
        raise ValueError(f"Missing placeholders in values_dict: {', '.join(missing_placeholders)}")
    
    # Create a copy of the template to modify
    result = html_template
    
    # Process each unique placeholder
    for placeholder in set(placeholders):
        # Get the value from dictionary
        replacement = values_dict[placeholder]
        
        # Escape HTML to prevent injection
        escaped_replacement = html.escape(str(replacement))
        
        # Replace all occurrences of this placeholder
        pattern = r'\{\{' + re.escape(placeholder) + r'\}\}'
        result = re.sub(pattern, escaped_replacement, result)
    
    return result
