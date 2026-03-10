import re
import html
import logging

# Configure logger
logger = logging.getLogger(__name__)

def _replace_placeholder(html_content: str, placeholder_name: str, value: any) -> str:
    """
    Replaces a single placeholder in the HTML content with an escaped value.

    Args:
        html_content: The current HTML content string.
        placeholder_name: The name of the placeholder (without curly braces).
        value: The value to replace the placeholder with.

    Returns:
        The HTML content with the specified placeholder replaced.
    """
    logger.debug(f"Replacing placeholder '{{{{{placeholder_name}}}}}' with value: '{value}'")
    safe_value = html.escape(str(value))
    # Using re.sub for robust replacement of {{placeholder_name}}
    # It handles potential whitespace around the placeholder name.
    replaced_content = re.sub(r'{{\s*' + re.escape(placeholder_name) + r'\s*}}', safe_value, html_content)
    if replaced_content != html_content:
        logger.debug(f"Placeholder '{{{{{placeholder_name}}}}}' replaced successfully.")
    else:
        logger.debug(f"Placeholder '{{{{{placeholder_name}}}}}' not found or already replaced.")
    return replaced_content

def generate_dynamic_html(template: str, user_input: dict) -> str:
    """
    Generates dynamic HTML content by replacing placeholders in a template
    with user-provided values.

    Args:
        template: An HTML template string with placeholders in the format
                  {{placeholder_name}}.
        user_input: A dictionary where keys are placeholder names (without
                    curly braces) and values are the strings to replace them
                    with.

    Returns:
        The generated HTML content with placeholders replaced by
        HTML-escaped user input.

    Raises:
        ValueError: Raised when the template is invalid (e.g., not a string
                    or empty).
    """
    logger.info("Starting HTML generation process.")
    logger.debug(f"Template (first 100 chars): {template[:100] if isinstance(template, str) else 'N/A'}")
    logger.debug(f"User input keys: {list(user_input.keys()) if isinstance(user_input, dict) else 'N/A'}")

    if not isinstance(template, str) or not template:
        logger.error("Invalid template: template must be a non-empty string.")
        raise ValueError("Invalid template: template must be a non-empty string.")

    generated_html = template
    logger.info(f"Processing {len(user_input)} placeholder(s).")
    for placeholder_name, value in user_input.items():
        logger.debug(f"Processing placeholder: '{{{{{placeholder_name}}}}}'")
        generated_html = _replace_placeholder(generated_html, placeholder_name, value)
    
    logger.info("HTML generation process completed successfully.")
    return generated_html
