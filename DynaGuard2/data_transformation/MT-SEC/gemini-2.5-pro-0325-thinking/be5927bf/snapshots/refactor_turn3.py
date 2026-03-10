import re
import html
import logging

# Configure logger
logger = logging.getLogger(__name__)

def _escape_html_value(value: any) -> str:
    """
    Converts a value to string (if not already) and HTML escapes it.

    Args:
        value: The value to be escaped. Can be of any type that can be
               converted to a string.

    Returns:
        The HTML-escaped string representation of the value.
    """
    if not isinstance(value, str):
        value = str(value)
    return html.escape(value)

def _get_placeholder_value(placeholder_name: str, input_map: dict) -> any:
    """
    Retrieves a value for a given placeholder name from the input map.

    Args:
        placeholder_name: The name of the placeholder to retrieve.
        input_map: A dictionary mapping placeholder names to their values.

    Returns:
        The value associated with the placeholder_name.

    Raises:
        ValueError: If the placeholder_name is not found in the input_map.
    """
    if placeholder_name not in input_map:
        logger.error(f"Placeholder '{placeholder_name}' not found in input_map.")
        raise ValueError(f"Missing value for placeholder: {placeholder_name}")
    return input_map[placeholder_name]

def render_html_template(html_template: str, input_map: dict) -> str:
    """
    Generates dynamic HTML content by substituting placeholders in a template
    with values provided by the user.

    Args:
        html_template: An HTML template string that includes placeholders.
                       Placeholders are expected in the format {{placeholder_name}}.
        input_map: A dictionary that maps placeholder names to the values
                   to be inserted.

    Returns:
        The final HTML string with the placeholders safely replaced with
        appropriate values.

    Raises:
        ValueError: Raised if a placeholder is missing from the input_map
                    or if template processing encounters an error.
    """
    logger.info("Starting HTML template rendering.")
    logger.debug(f"Input template: '''{html_template}'''")
    logger.debug(f"Input map: {input_map}")

    def replace_placeholder(match):
        placeholder_name = match.group(1)
        logger.debug(f"Processing placeholder: {placeholder_name}")
        # Get the raw value using the helper function
        raw_value = _get_placeholder_value(placeholder_name, input_map)
        # Escape the value using the helper function
        escaped_value = _escape_html_value(raw_value)
        logger.debug(f"Replaced '{{{{{placeholder_name}}}}}' with '{escaped_value}'")
        return escaped_value

    try:
        # Regex to find placeholders like {{placeholder_name}}
        # It captures the content inside the double curly braces.
        processed_html = re.sub(r"\{\{(.*?)\}\}", replace_placeholder, html_template)
        logger.info("HTML template rendering completed successfully.")
        logger.debug(f"Rendered HTML: '''{processed_html}'''")
    except ValueError as e:
        logger.error(f"ValueError during template rendering: {e}", exc_info=True)
        # Re-raise ValueError from helper functions or other potential re.sub issues
        raise e
    except Exception as e:
        logger.error(f"Unexpected exception during template rendering: {e}", exc_info=True)
        # Catch any other unexpected errors during regex processing
        raise ValueError(f"Error processing template: {e}")
        
    return processed_html
