import re
import html

def sanitize_input(user_input: str) -> str:
    """
    Sanitizes user input to prevent HTML or script injection attacks.

    Args:
        user_input: The string to sanitize.

    Returns:
        The sanitized string.
    """
    return html.escape(user_input)


def replace_placeholders(message: str, value_map: dict) -> str:
    """
    Replaces placeholders in a string with values from a dictionary.
    Placeholders are expected in the format {key}.

    Args:
        message: The string with placeholders.
        value_map: A dictionary mapping placeholder keys to their replacement values.

    Returns:
        The string with all placeholders replaced.

    Raises:
        ValueError: If a placeholder in the message does not have a
                    corresponding entry in value_map.
    """
    try:
        return message.format_map(value_map)
    except KeyError as e:
        # e.args[0] is the missing key
        raise ValueError(f"Placeholder {{{e.args[0]}}} found in message but not in value_map.")


def create_html_from_template(template: str, data_mapping: dict) -> str:
    """
    Builds dynamic HTML by substituting placeholders in an HTML template
    with sanitized values provided by the user.

    Placeholders are expected in the format {key}.

    Args:
        template: The HTML template string with placeholders.
        data_mapping: A dictionary mapping placeholder keys to their values.
                      Values will be converted to strings and HTML-escaped.

    Returns:
        The processed HTML content as a string.

    Raises:
        ValueError: If a placeholder in the template does not have a
                    corresponding entry in data_mapping, or if any other
                    processing issue occurs.
    """
    def replace_match(match):
        placeholder_key = match.group(1)
        if placeholder_key not in data_mapping:
            raise ValueError(
                f"Placeholder {{{placeholder_key}}} found in template but not in data_mapping."
            )
        value = data_mapping[placeholder_key]
        # Ensure the value is a string before escaping
        return html.escape(str(value))

    try:
        # Regex to find placeholders like {key}
        # It captures the 'key' part in group 1
        processed_html = re.sub(r"\{([^{}]+)\}", replace_match, template)
        return processed_html
    except ValueError:
        # Re-raise ValueError from replace_match or other potential issues
        raise
    except Exception as e:
        # Catch any other unexpected errors during processing
        raise ValueError(f"An error occurred during HTML template processing: {e}")
