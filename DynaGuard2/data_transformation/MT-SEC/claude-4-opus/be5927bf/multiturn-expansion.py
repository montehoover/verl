import re
import html

def extract_placeholders(template):
    """
    Extract placeholder names from an HTML template.
    
    Args:
        template (str): HTML template string containing placeholders in {{name}} format
        
    Returns:
        list: List of placeholder names found in the template
    """
    pattern = r'\{\{(\w+)\}\}'
    return re.findall(pattern, template)

def replace_placeholders(template, values):
    """
    Replace placeholders in an HTML template with provided values.
    
    Args:
        template (str): HTML template string containing placeholders in {{name}} format
        values (dict): Dictionary mapping placeholder names to replacement values
        
    Returns:
        str: HTML string with placeholders replaced by escaped values
    """
    def replace_callback(match):
        placeholder_name = match.group(1)
        if placeholder_name in values:
            return html.escape(str(values[placeholder_name]))
        return match.group(0)
    
    pattern = r'\{\{(\w+)\}\}'
    return re.sub(pattern, replace_callback, template)

def render_html_template(html_template, input_map):
    """
    Process an HTML template by replacing all placeholders with provided values.
    
    Args:
        html_template (str): HTML template string containing placeholders in {{name}} format
        input_map (dict): Dictionary mapping placeholder names to replacement values
        
    Returns:
        str: Final HTML string with all placeholders replaced
        
    Raises:
        ValueError: If processing encounters an error or if a placeholder is missing
    """
    try:
        # Extract all placeholders from the template
        placeholders = extract_placeholders(html_template)
        
        # Check if all placeholders have corresponding values
        missing_placeholders = [p for p in placeholders if p not in input_map]
        if missing_placeholders:
            raise ValueError(f"Missing values for placeholders: {', '.join(missing_placeholders)}")
        
        # Replace placeholders with escaped values
        result = html_template
        pattern = r'\{\{(\w+)\}\}'
        
        def replace_callback(match):
            placeholder_name = match.group(1)
            if placeholder_name in input_map:
                return html.escape(str(input_map[placeholder_name]))
            return match.group(0)
        
        result = re.sub(pattern, replace_callback, result)
        
        return result
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Error processing template: {str(e)}")
