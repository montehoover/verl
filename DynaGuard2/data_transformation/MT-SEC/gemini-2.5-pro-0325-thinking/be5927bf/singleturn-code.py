import re
import html

def render_html_template(html_template: str, input_map: dict) -> str:
    """
    Generates dynamic HTML content by substituting placeholders in a template
    with values provided by the user. Handles each placeholder securely.

    Args:
        html_template: An HTML template string that includes placeholders
                       (e.g., "{{username}}").
        input_map: A dictionary that maps placeholder names to the values
                   to be inserted.

    Returns:
        The final HTML string with the placeholders safely replaced with
        appropriate values.

    Raises:
        ValueError: Raised if a placeholder in the template is missing from
                    the input_map or if template processing encounters an error.
    """
    def replace_placeholder(match):
        placeholder_name = match.group(1).strip()
        if placeholder_name not in input_map:
            raise ValueError(f"Placeholder '{{{{{placeholder_name}}}}}' not found in input_map.")
        
        value = input_map[placeholder_name]
        # Ensure the value is a string before escaping, as html.escape expects a string
        escaped_value = html.escape(str(value))
        return escaped_value

    try:
        # Regex to find placeholders like {{ placeholder_name }}
        # It captures the content between {{ and }}
        processed_html = re.sub(r"\{\{(.*?)\}\}", replace_placeholder, html_template)
    except ValueError as e:
        # Re-raise ValueError from replace_placeholder or other potential regex errors
        raise e
    except Exception as e:
        # Catch any other unexpected errors during regex processing
        raise ValueError(f"Template processing encountered an error: {e}") from e
        
    # Final check to ensure no {{...}} are left if re.sub didn't raise an error
    # This can happen if a placeholder was malformed in a way re.sub skipped it but it's still present
    if re.search(r"\{\{(.*?)\}\}", processed_html):
        # This case should ideally be caught by the placeholder check,
        # but as a safeguard if some placeholders were not processed correctly.
        # However, the current logic with re.sub and replace_placeholder should handle all valid {{...}}
        # or raise ValueError if a key is missing.
        # This check is more for unexpected template structures.
        # For this problem, if a placeholder was in the template but not in input_map,
        # replace_placeholder would have already raised ValueError.
        # So, this specific check might be redundant if all placeholders are well-formed.
        # Let's assume for now that if re.sub completes without error, all placeholders were processed.
        pass

    return processed_html
