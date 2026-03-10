import re


def check_javascript_scheme(url_str: str) -> bool:
    """
    Determines whether or not a given string contains a potentially harmful URL scheme
    (e.g., JavaScript, vbscript, etc.), while excluding image-based data URLs from
    this evaluation.

    Args:
        url_str: The string in question to be examined for potentially
                 dangerous URL schemes.

    Returns:
        True if the input string contains a dangerous URL scheme (except safe image
        data URLs); returns False otherwise.

    Raises:
        re.error: Incorrectly formatted inputs or regex patterns could raise
                  're.error' exceptions.
    """
    # Regex to identify safe image data URLs (e.g., data:image/png;base64,...)
    # We want to allow these, so if it matches, it's not harmful in this context.
    safe_image_data_url_pattern = re.compile(r"^\s*data:image\/", re.IGNORECASE)

    if safe_image_data_url_pattern.match(url_str):
        return False

    # Regex to identify potentially harmful URL schemes
    # Looks for common script-related schemes at the beginning of the string.
    # \s* allows for leading whitespace.
    # (?:...) is a non-capturing group.
    # re.IGNORECASE makes the match case-insensitive.
    harmful_scheme_pattern = re.compile(
        r"^\s*(?:javascript|vbscript|script|about|data):", re.IGNORECASE
    )

    # Check if the URL matches any of the harmful schemes.
    if harmful_scheme_pattern.match(url_str):
        # If it's not an image data URL (checked above) but matches 'data:',
        # it could be other types of data URLs that might be harmful (e.g., data:text/html).
        # So, we consider it harmful here.
        return True

    return False
