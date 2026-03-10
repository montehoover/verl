import re
import html

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
    safe_value = html.escape(str(value))
    # Using re.sub for robust replacement of {{placeholder_name}}
    # It handles potential whitespace around the placeholder name.
    return re.sub(r'{{\s*' + re.escape(placeholder_name) + r'\s*}}', safe_value, html_content)

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
    if not isinstance(template, str) or not template:
        raise ValueError("Invalid template: template must be a non-empty string.")

    generated_html = template
    for placeholder_name, value in user_input.items():
        generated_html = _replace_placeholder(generated_html, placeholder_name, value)
    
    return generated_html
