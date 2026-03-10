import re

# Pre-compiled regular expression for potentially malicious URL schemes.
# Looks for "javascript:", "vbscript:", or "data:" schemes.
# The \b ensures that 'data' is a whole word, preventing false positives on strings like "mydata:".
_MALICIOUS_URL_SCHEME_PATTERN = re.compile(
    r'\b(?:javascript|vbscript|data):',
    re.IGNORECASE
)

# Pre-compiled regular expression for valid image data URLs.
# Must start with "data:image/" followed by a recognized image subtype,
# then any characters except a comma, and finally a comma.
_VALID_IMAGE_DATA_URL_PATTERN = re.compile(
    r'^data:image/(?:png|jpeg|jpg|gif|bmp|webp|svg\+xml)'  # Common image subtypes
    r'[^,]*,',  # Allows any characters except a comma, followed by a comma
    re.IGNORECASE
)


def detect_malicious_url_scheme(url_string: str) -> bool:
    """
    Determines whether a given string contains a risky URL scheme,
    such as javascript, vbscript, or a generic data: scheme,
    while excluding valid image data URLs from being flagged.

    Args:
        url_string: The input string to be inspected for potentially
                    malicious URL schemes.

    Returns:
        True if the string contains any dangerous URL schemes (and is not
        a valid image data URL); otherwise, False.
        Possible exceptions like 're.error' could occur with malformed
        regex patterns, though not raised explicitly by this function.
    """
    # Check if the URL string matches the pattern for a valid (and safe) image data URL.
    # If it is a valid image data URL, it's not considered malicious in this context.
    if _VALID_IMAGE_DATA_URL_PATTERN.match(url_string):
        return False

    # If it's not a safe image data URL, check if it contains other malicious schemes.
    # This includes "javascript:", "vbscript:", or "data:" (when not a safe image).
    if _MALICIOUS_URL_SCHEME_PATTERN.search(url_string):
        return True

    # If none of the above conditions are met, the URL scheme is not considered malicious.
    return False
