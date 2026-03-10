import re
import html
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)


def extract_placeholders(html_template: str) -> list:
    """
    Extract all placeholder names from the HTML template.
    
    This function uses regular expressions to find all placeholders 
    in the template that are delimited by double curly braces {{...}}.
    
    Args:
        html_template (str): HTML template containing placeholders 
            delimited with {{...}}
        
    Returns:
        list: List of placeholder names found in the template
        
    Example:
        >>> extract_placeholders("Hello {{name}}, welcome to {{site}}!")
        ['name', 'site']
    """
    logger.debug(f"Extracting placeholders from template of length {len(html_template)}")
    placeholders = re.findall(r'\{\{([^}]+)\}\}', html_template)
    logger.info(f"Found {len(placeholders)} placeholders: {placeholders}")
    return placeholders


def validate_placeholders(placeholders: list, user_values: dict) -> None:
    """
    Validate that all placeholders have corresponding values.
    
    This function checks if each placeholder found in the template
    has a corresponding key in the user_values dictionary.
    
    Args:
        placeholders (list): List of placeholder names from the template
        user_values (dict): Mapping of placeholder names to values
        
    Raises:
        ValueError: If a placeholder is missing from user_values
        
    Example:
        >>> validate_placeholders(['name', 'age'], {'name': 'John', 'age': 30})
        None
        >>> validate_placeholders(['name', 'age'], {'name': 'John'})
        ValueError: Missing value for placeholder: {{age}}
    """
    logger.debug(f"Validating {len(placeholders)} placeholders against {len(user_values)} user values")
    
    for placeholder in placeholders:
        if placeholder not in user_values:
            error_msg = f"Missing value for placeholder: {{{{{placeholder}}}}}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    logger.info("All placeholders validated successfully")


def insert_values(html_template: str, user_values: dict) -> str:
    """
    Replace placeholders in the template with escaped user values.
    
    This function iterates through all user values and replaces their
    corresponding placeholders in the template. Values are HTML-escaped
    to prevent XSS attacks.
    
    Args:
        html_template (str): HTML template containing placeholders
        user_values (dict): Mapping of placeholder names to values
        
    Returns:
        str: HTML content with placeholders replaced by escaped values
        
    Example:
        >>> template = "Hello {{name}}, your score is {{score}}"
        >>> values = {'name': 'Alice', 'score': 100}
        >>> insert_values(template, values)
        'Hello Alice, your score is 100'
    """
    logger.debug(f"Inserting {len(user_values)} values into template")
    result = html_template
    
    for placeholder, value in user_values.items():
        # Escape the value for safe HTML insertion
        escaped_value = html.escape(str(value))
        logger.debug(f"Replacing placeholder '{{{{placeholder}}}}' with '{escaped_value}'")
        
        # Replace all occurrences of this placeholder
        pattern = r'\{\{' + re.escape(placeholder) + r'\}\}'
        result = re.sub(pattern, escaped_value, result)
    
    logger.info("Successfully inserted all values into template")
    return result


def create_html_template(html_template: str, user_values: dict) -> str:
    """
    Produce dynamic HTML content by replacing placeholders in a template with user-provided values.
    
    This is the main function that orchestrates the template processing workflow:
    1. Extracts placeholders from the template
    2. Validates that all placeholders have corresponding values
    3. Inserts the values into the template with proper HTML escaping
    
    Args:
        html_template (str): HTML template containing placeholders delimited with {{...}}
        user_values (dict): Mapping of placeholder names to values to be inserted
        
    Returns:
        str: HTML content generated after placeholder replacement
        
    Raises:
        ValueError: When any error occurs during template processing or if a placeholder is missing
        
    Example:
        >>> template = "<h1>Welcome {{username}}!</h1><p>Your email: {{email}}</p>"
        >>> values = {'username': 'John Doe', 'email': 'john@example.com'}
        >>> create_html_template(template, values)
        '<h1>Welcome John Doe!</h1><p>Your email: john@example.com</p>'
    """
    logger.info("Starting HTML template processing")
    
    try:
        # Extract placeholders from the template
        placeholders = extract_placeholders(html_template)
        
        # Validate all placeholders have values
        validate_placeholders(placeholders, user_values)
        
        # Insert values into the template
        result = insert_values(html_template, user_values)
        
        logger.info("HTML template processing completed successfully")
        return result
        
    except Exception as e:
        if isinstance(e, ValueError):
            logger.error(f"Validation error: {str(e)}")
            raise
        else:
            error_msg = f"Error processing template: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
