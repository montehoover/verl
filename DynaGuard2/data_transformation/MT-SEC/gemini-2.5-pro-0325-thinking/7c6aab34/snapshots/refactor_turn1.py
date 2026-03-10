import re
import html

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
        # Ensure value is a string before escaping
        safe_value = html.escape(str(value))
        # Using re.sub for robust replacement of {{placeholder_name}}
        # It's generally safer than multiple str.replace if placeholders could be substrings of each other
        # or if we need more complex matching rules in the future.
        # For simple {{key}} replacement, str.replace is also fine.
        generated_html = re.sub(r'{{\s*' + re.escape(placeholder_name) + r'\s*}}', safe_value, generated_html)
    
    return generated_html
