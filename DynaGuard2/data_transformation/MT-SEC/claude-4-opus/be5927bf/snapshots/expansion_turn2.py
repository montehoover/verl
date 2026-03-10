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
