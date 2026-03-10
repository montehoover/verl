import re
import html

def create_html_template(html_template: str, user_values: dict) -> str:
    """
    Produces dynamic HTML content by replacing placeholders in a given template
    with corresponding user-provided values.

    Args:
        html_template: An HTML template string containing placeholders
                       delimited with {{...}}.
        user_values: A mapping of placeholder names to the values to be inserted.

    Returns:
        The HTML content generated after placeholder replacement.

    Raises:
        ValueError: Raised when any error occurs during template processing
                    or if a placeholder is missing.
    """
    try:
        # Find all placeholders like {{placeholder_name}}
        placeholders = re.findall(r"\{\{(.*?)\}\}", html_template)

        processed_template = html_template

        for placeholder_key in placeholders:
            placeholder_full = f"{{{{{placeholder_key}}}}}"
            if placeholder_key not in user_values:
                raise ValueError(f"Missing value for placeholder: {placeholder_key}")

            value = user_values[placeholder_key]
            escaped_value = html.escape(str(value))
            processed_template = processed_template.replace(placeholder_full, escaped_value)
        
        # Check if any {{...}} are left, which might indicate an issue or malformed placeholder
        # that wasn't caught by the initial findall if it was, for example, {{placeholder_name }}.
        # The current regex r"\{\{(.*?)\}\}" is quite greedy for the content, so this might be redundant
        # but can catch edge cases like empty placeholders {{}} if they are not intended to be processed.
        # However, the problem asks to raise ValueError if a *placeholder is missing* from user_values,
        # which is already handled. If there are {{}} left that were not in user_values,
        # they would have been caught. If they were malformed and not caught by findall,
        # they would just remain in the string.
        # For now, let's stick to the primary requirement of replacing found placeholders.

        return processed_template

    except ValueError as ve: # Re-raise ValueError as is
        raise ve
    except Exception as e: # Catch any other unexpected errors
        raise ValueError(f"An error occurred during template processing: {e}")
