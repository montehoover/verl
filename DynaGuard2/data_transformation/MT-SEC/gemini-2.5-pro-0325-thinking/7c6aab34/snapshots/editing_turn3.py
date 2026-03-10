import re
import html

def generate_dynamic_html(template: str, user_input: dict) -> str:
    """
    Generates HTML content by replacing placeholders in a template with user-provided values.

    Placeholders are expected in the format {key}.
    User-provided values are HTML-escaped before substitution to prevent XSS.

    Args:
        template: A string containing HTML content with placeholders.
        user_input: A dictionary where keys are placeholder names and values are
                    the content to substitute. Values will be HTML-escaped.

    Returns:
        The generated HTML content with placeholders replaced by escaped user input.

    Raises:
        ValueError: If the template is invalid (e.g., unmatched braces),
                    a placeholder is missing a corresponding value in user_input,
                    or any other processing error occurs.
    """
    try:
        # Check for basic malformations like unmatched braces.
        if template.count('{') != template.count('}'):
            raise ValueError("Invalid template: Unmatched braces.")

        # HTML-escape all string values from user_input.
        # Non-string values are converted to string first, then escaped.
        escaped_user_input = {
            k: html.escape(str(v)) for k, v in user_input.items()
        }

        # Attempt to format the string using the escaped user input.
        # str.format_map will raise a KeyError if a placeholder in the template
        # is not found in escaped_user_input.
        generated_html = template.format_map(escaped_user_input)
        
        # An additional check for any remaining unescaped placeholders like "{key}"
        # could be done here if necessary, though format_map should handle missing keys.
        # This might be relevant if template contains "{{" or "}}" which are
        # literal braces after formatting.
        # For example, re.search(r"(?<!{){[^{}]+}(?!})", generated_html)
        # could find unescaped single placeholders, but format_map's KeyError is usually sufficient.

        return generated_html
    except KeyError as e:
        # This occurs if a key in the template string (e.g., "{my_key}")
        # is not found in user_input.
        raise ValueError(f"Invalid template: Missing value for placeholder {e}.") from e
    except ValueError as e:
        # This can re-raise our custom ValueError or catch others from format_map
        # for certain template string issues.
        raise ValueError(f"Invalid template or processing error: {e}") from e
    except Exception as e:
        # Catch any other unexpected errors during formatting.
        raise ValueError(f"An unexpected error occurred during HTML generation: {e}") from e
