import re
import html
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)


def parse_placeholders(template: str) -> list:
    """
    Extracts placeholder names from the template.
    
    This function searches for placeholders in the format {{placeholder_name}}
    within the provided template string. It validates that placeholders follow
    the expected format and raises an error if invalid syntax is detected.
    
    Args:
        template (str): HTML template with placeholders in the format {{placeholder_name}}.
                       Placeholder names must contain only word characters (letters,
                       digits, and underscores).
        
    Returns:
        list: A list of placeholder names (strings) found in the template.
              Returns an empty list if no placeholders are found.
        
    Raises:
        ValueError: When the template contains '{{' but has invalid placeholder
                   format (e.g., missing closing braces, invalid characters).
    
    Example:
        >>> parse_placeholders("<h1>{{title}}</h1><p>{{content}}</p>")
        ['title', 'content']
    """
    logger.debug(f"Parsing placeholders from template of length {len(template)}")
    
    placeholder_pattern = r'\{\{(\w+)\}\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    # Check if template has valid placeholders
    if not placeholders and '{{' in template:
        logger.error("Invalid placeholder format detected in template")
        raise ValueError("Invalid placeholder format in template")
    
    logger.info(f"Found {len(placeholders)} placeholders: {placeholders}")
    return placeholders


def replace_placeholder(template: str, placeholder: str, value: str) -> str:
    """
    Replaces a single placeholder in the template with the escaped value.
    
    This function performs a safe replacement of a placeholder with user-provided
    content. The value is HTML-escaped to prevent XSS attacks and other security
    vulnerabilities.
    
    Args:
        template (str): HTML template containing the placeholder to be replaced.
        placeholder (str): The placeholder name to replace (without the curly braces).
        value (str): The value to insert in place of the placeholder. This value
                    will be converted to string and HTML-escaped for security.
        
    Returns:
        str: The template with the specified placeholder replaced by the escaped value.
             If the placeholder is not found, returns the original template unchanged.
    
    Example:
        >>> replace_placeholder("<h1>{{title}}</h1>", "title", "<script>alert('XSS')</script>")
        "<h1>&lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;</h1>"
    """
    logger.debug(f"Replacing placeholder '{placeholder}' with value of length {len(str(value))}")
    
    escaped_value = html.escape(str(value))
    result = template.replace(f'{{{{{placeholder}}}}}', escaped_value)
    
    if result != template:
        logger.debug(f"Successfully replaced placeholder '{placeholder}'")
    else:
        logger.warning(f"Placeholder '{placeholder}' not found in template")
    
    return result


def generate_dynamic_html(template: str, user_input: dict) -> str:
    """
    Generates dynamic HTML content based on user input and a template.
    
    This function processes an HTML template containing placeholders and replaces
    them with user-provided values. It ensures security by HTML-escaping all
    user input to prevent XSS attacks. Only placeholders that have corresponding
    values in the user_input dictionary will be replaced; others remain unchanged.
    
    Args:
        template (str): HTML template with placeholders in the format {{placeholder_name}}.
                       The template must be a valid string.
        user_input (dict): A dictionary mapping placeholder names (keys) to their
                          replacement values. Values can be of any type and will
                          be converted to strings before insertion.
        
    Returns:
        str: The generated HTML content with placeholders replaced by the
             corresponding user-provided values. Placeholders without matching
             keys in user_input remain unchanged in the output.
        
    Raises:
        ValueError: Raised in the following cases:
                   - When template is not a string
                   - When user_input is not a dictionary
                   - When the template contains invalid placeholder syntax
    
    Example:
        >>> template = "<h1>{{title}}</h1><p>Welcome, {{name}}!</p>"
        >>> user_data = {"title": "My Page", "name": "Alice"}
        >>> generate_dynamic_html(template, user_data)
        "<h1>My Page</h1><p>Welcome, Alice!</p>"
    """
    logger.info("Starting HTML generation process")
    
    if not isinstance(template, str):
        logger.error(f"Invalid template type: {type(template)}")
        raise ValueError("Template must be a string")
    
    if not isinstance(user_input, dict):
        logger.error(f"Invalid user_input type: {type(user_input)}")
        raise ValueError("User input must be a dictionary")
    
    logger.debug(f"Processing template with {len(user_input)} user inputs")
    
    try:
        # Parse placeholders from template
        placeholders = parse_placeholders(template)
        
        # Replace each placeholder with the corresponding value
        result = template
        replaced_count = 0
        
        for placeholder in placeholders:
            if placeholder in user_input:
                result = replace_placeholder(result, placeholder, user_input[placeholder])
                replaced_count += 1
            else:
                logger.warning(f"No value provided for placeholder '{placeholder}'")
        
        logger.info(f"Successfully replaced {replaced_count} out of {len(placeholders)} placeholders")
        return result
        
    except Exception as e:
        logger.error(f"Error during HTML generation: {str(e)}")
        raise
