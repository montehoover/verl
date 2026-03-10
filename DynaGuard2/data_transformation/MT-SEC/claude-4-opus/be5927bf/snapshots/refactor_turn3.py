import re
import html
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)


def find_placeholders(html_template: str) -> list:
    """
    Extract all placeholder names from the HTML template.
    
    This function uses regular expressions to identify placeholders 
    in the format {{placeholder_name}} within the HTML template.
    
    Args:
        html_template (str): An HTML template string that includes 
            placeholders in the format {{placeholder_name}}.
        
    Returns:
        list: A list of placeholder names (strings) found in the template.
            Returns an empty list if no placeholders are found.
            
    Example:
        >>> template = "<h1>Hello {{name}}, welcome to {{site}}!</h1>"
        >>> find_placeholders(template)
        ['name', 'site']
    """
    placeholder_pattern = r'\{\{(\w+)\}\}'
    placeholders = re.findall(placeholder_pattern, html_template)
    logger.debug(f"Found placeholders: {placeholders}")
    return placeholders


def validate_placeholders(placeholders: list, input_map: dict) -> None:
    """
    Validate that all placeholders have corresponding values in the input map.
    
    This function ensures that every placeholder found in the template
    has a corresponding key in the input_map dictionary. This prevents
    incomplete template rendering.
    
    Args:
        placeholders (list): List of placeholder names extracted from 
            the template.
        input_map (dict): Dictionary mapping placeholder names to their
            replacement values.
        
    Raises:
        ValueError: If any placeholder is missing from the input_map.
        
    Example:
        >>> placeholders = ['name', 'age']
        >>> input_map = {'name': 'John', 'age': 25}
        >>> validate_placeholders(placeholders, input_map)  # No error
        
        >>> input_map = {'name': 'John'}
        >>> validate_placeholders(placeholders, input_map)
        ValueError: Missing value for placeholder: age
    """
    for placeholder in placeholders:
        if placeholder not in input_map:
            logger.error(f"Missing value for placeholder: {placeholder}")
            raise ValueError(f"Missing value for placeholder: {placeholder}")
    logger.debug("All placeholders validated successfully")


def escape_html_value(value: any) -> str:
    """
    Safely escape a value for HTML insertion to prevent XSS attacks.
    
    This function converts any input value to a string and then escapes
    special HTML characters to prevent cross-site scripting (XSS) attacks.
    
    Args:
        value (any): The value to be escaped. Can be of any type as it
            will be converted to string before escaping.
        
    Returns:
        str: HTML-escaped string representation of the value.
        
    Example:
        >>> escape_html_value("<script>alert('XSS')</script>")
        '&lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;'
        
        >>> escape_html_value(123)
        '123'
    """
    escaped = html.escape(str(value))
    logger.debug(f"Escaped value: {repr(value)} -> {repr(escaped)}")
    return escaped


def substitute_placeholder(template: str, placeholder: str, value: str) -> str:
    """
    Replace a single placeholder in the template with the given value.
    
    This function replaces all occurrences of a specific placeholder
    (in the format {{placeholder_name}}) with the provided value.
    
    Args:
        template (str): The HTML template string containing placeholders.
        placeholder (str): The name of the placeholder to replace 
            (without the curly braces).
        value (str): The escaped value to insert in place of the 
            placeholder.
        
    Returns:
        str: The template with all occurrences of the specified 
            placeholder replaced with the value.
            
    Example:
        >>> template = "<p>Hello {{name}}, {{name}} is awesome!</p>"
        >>> substitute_placeholder(template, "name", "Alice")
        '<p>Hello Alice, Alice is awesome!</p>'
    """
    pattern = r'\{\{' + re.escape(placeholder) + r'\}\}'
    result = re.sub(pattern, value, template)
    logger.debug(f"Substituted placeholder '{placeholder}' with value")
    return result


def render_html_template(html_template: str, input_map: dict) -> str:
    """
    Generates dynamic HTML content by substituting placeholders in a template
    with values provided by the user.
    
    This is the main function that orchestrates the template rendering process.
    It finds all placeholders in the template, validates that all required
    values are provided, escapes the values for safe HTML insertion, and
    performs the substitution.
    
    The function handles placeholders in the format {{placeholder_name}} where
    placeholder_name can contain alphanumeric characters and underscores.
    
    Args:
        html_template (str): An HTML template string that includes 
            placeholders in the format {{placeholder_name}}.
        input_map (dict): A dictionary that maps placeholder names 
            (as strings) to the values to be inserted. Values can be 
            of any type and will be converted to strings and escaped.
        
    Returns:
        str: The final HTML string with all placeholders safely replaced 
            with their corresponding values from input_map.
        
    Raises:
        ValueError: Raised if the template processing encounters an error 
            or if any placeholder in the template is missing from the 
            input_map.
            
    Example:
        >>> template = '''
        ... <html>
        ...   <body>
        ...     <h1>Welcome {{username}}!</h1>
        ...     <p>Your score is: {{score}}</p>
        ...   </body>
        ... </html>
        ... '''
        >>> values = {'username': 'John Doe', 'score': 100}
        >>> result = render_html_template(template, values)
        >>> # Result will have placeholders replaced with escaped values
    """
    try:
        logger.info("Starting template rendering")
        logger.debug(f"Input template: {html_template}")
        logger.debug(f"Input map: {input_map}")
        
        # Find all placeholders in the template
        placeholders = find_placeholders(html_template)
        
        # Validate all placeholders have values
        validate_placeholders(placeholders, input_map)
        
        # Replace each placeholder with escaped value
        result = html_template
        for placeholder, value in input_map.items():
            escaped_value = escape_html_value(value)
            result = substitute_placeholder(result, placeholder, escaped_value)
        
        logger.info("Template rendering completed successfully")
        logger.debug(f"Rendered HTML: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Template processing error: {str(e)}")
        raise ValueError(f"Template processing error: {str(e)}")
