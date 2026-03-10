import re

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
