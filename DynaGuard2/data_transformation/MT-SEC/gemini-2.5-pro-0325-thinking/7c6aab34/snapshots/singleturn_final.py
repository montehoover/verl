import re
import html

def generate_dynamic_html(template: str, user_input: dict) -> str:
    """
    Generates dynamic HTML content by replacing placeholders in a template
    with user-provided values.

    Args:
        template: An HTML template string with placeholders in the format {{key}}.
        user_input: A dictionary where keys are placeholder names (without curly braces)
                    and values are the strings to insert.

    Returns:
        The generated HTML content with placeholders replaced by user-provided,
        HTML-escaped values.

    Raises:
        ValueError: Raised if the template is not a string.
    """
    if not isinstance(template, str):
        raise ValueError("Invalid template: template must be a string.")

    processed_html = template
    for key, value in user_input.items():
        placeholder = f"{{{{{key}}}}}"
        # Ensure value is a string before escaping, as html.escape expects a string
        escaped_value = html.escape(str(value))
        processed_html = processed_html.replace(placeholder, escaped_value)
    
    return processed_html
