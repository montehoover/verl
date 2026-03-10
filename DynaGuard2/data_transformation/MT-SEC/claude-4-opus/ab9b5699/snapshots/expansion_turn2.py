import re
import html

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

def replace_placeholders(template, values):
    """
    Replaces placeholders in an HTML template with provided values.
    
    Args:
        template (str): HTML template string containing placeholders in {{...}} format
        values (dict): Dictionary mapping placeholder names to their values
        
    Returns:
        str: HTML content with placeholders replaced by escaped values
    """
    # Function to replace each match
    def replacer(match):
        placeholder = match.group(1).strip()
        if placeholder in values:
            # Escape the value to prevent HTML injection
            return html.escape(str(values[placeholder]))
        else:
            # Keep the original placeholder if no value provided
            return match.group(0)
    
    # Replace all placeholders
    pattern = r'\{\{([^}]+)\}\}'
    result = re.sub(pattern, replacer, template)
    
    return result
