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

def create_html_template(html_template, user_values):
    """
    Processes an HTML template by replacing placeholders with user-provided values.
    
    Args:
        html_template (str): HTML template string containing placeholders in {{...}} format
        user_values (dict): Dictionary mapping placeholder names to their values
        
    Returns:
        str: Generated HTML content with all placeholders replaced
        
    Raises:
        ValueError: If any error occurs during processing or if a placeholder is missing
    """
    try:
        # Extract all placeholders from the template
        placeholders = extract_placeholders(html_template)
        
        # Check if all placeholders have corresponding values
        missing_placeholders = []
        for placeholder in placeholders:
            if placeholder not in user_values:
                missing_placeholders.append(placeholder)
        
        if missing_placeholders:
            raise ValueError(f"Missing values for placeholders: {', '.join(missing_placeholders)}")
        
        # Replace placeholders with values
        result = replace_placeholders(html_template, user_values)
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error processing template: {str(e)}")
