import re
import html

def extract_placeholders(html_template: str) -> list:
    """
    Extract all placeholder names from the HTML template.
    
    Args:
        html_template: HTML template containing placeholders delimited with {{...}}
        
    Returns:
        List of placeholder names found in the template
    """
    return re.findall(r'\{\{([^}]+)\}\}', html_template)


def validate_placeholders(placeholders: list, user_values: dict) -> None:
    """
    Validate that all placeholders have corresponding values.
    
    Args:
        placeholders: List of placeholder names from the template
        user_values: Mapping of placeholder names to values
        
    Raises:
        ValueError: If a placeholder is missing from user_values
    """
    for placeholder in placeholders:
        if placeholder not in user_values:
            raise ValueError(f"Missing value for placeholder: {{{{{placeholder}}}}}")


def insert_values(html_template: str, user_values: dict) -> str:
    """
    Replace placeholders in the template with escaped user values.
    
    Args:
        html_template: HTML template containing placeholders
        user_values: Mapping of placeholder names to values
        
    Returns:
        HTML content with placeholders replaced by escaped values
    """
    result = html_template
    for placeholder, value in user_values.items():
        # Escape the value for safe HTML insertion
        escaped_value = html.escape(str(value))
        # Replace all occurrences of this placeholder
        pattern = r'\{\{' + re.escape(placeholder) + r'\}\}'
        result = re.sub(pattern, escaped_value, result)
    return result


def create_html_template(html_template: str, user_values: dict) -> str:
    """
    Produce dynamic HTML content by replacing placeholders in a template with user-provided values.
    
    Args:
        html_template: HTML template containing placeholders delimited with {{...}}
        user_values: Mapping of placeholder names to values to be inserted
        
    Returns:
        HTML content generated after placeholder replacement
        
    Raises:
        ValueError: When any error occurs during template processing or if a placeholder is missing
    """
    try:
        # Extract placeholders from the template
        placeholders = extract_placeholders(html_template)
        
        # Validate all placeholders have values
        validate_placeholders(placeholders, user_values)
        
        # Insert values into the template
        result = insert_values(html_template, user_values)
        
        return result
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Error processing template: {str(e)}")
