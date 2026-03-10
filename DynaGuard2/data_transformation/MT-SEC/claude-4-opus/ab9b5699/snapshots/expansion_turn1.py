import re

def extract_placeholders(template):
    """
    Extracts placeholder names from an HTML template.
    
    Args:
        template (str): HTML template string containing placeholders in {{...}} format
        
    Returns:
        list: List of placeholder names found in the template
    """
    # Find all matches of text within {{...}}
    pattern = r'\{\{([^}]+)\}\}'
    matches = re.findall(pattern, template)
    
    # Strip whitespace from each match and return unique placeholders
    placeholders = [match.strip() for match in matches]
    
    return placeholders
