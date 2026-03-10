import re
import html

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
    def replace_placeholder(match):
        placeholder_name = match.group(1)
        if placeholder_name not in input_map:
            raise ValueError(f"Missing value for placeholder: {placeholder_name}")
        
        value = input_map[placeholder_name]
        # Ensure all values are strings before escaping, or handle non-string types
        if not isinstance(value, str):
            value = str(value)
            
        return html.escape(value)

    try:
        # Regex to find placeholders like {{placeholder_name}}
        # It captures the content inside the double curly braces.
        processed_html = re.sub(r"\{\{(.*?)\}\}", replace_placeholder, html_template)
    except ValueError as e:
        # Re-raise ValueError from replace_placeholder or other potential re.sub issues
        raise e
    except Exception as e:
        # Catch any other unexpected errors during regex processing
        raise ValueError(f"Error processing template: {e}")
        
    return processed_html
