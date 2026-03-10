import re

"""
This module provides a function to detect potentially malicious URL schemes
in strings, with specific exceptions for image data URLs.
"""

# Pre-compiled regular expressions for efficiency and clarity

# Matches common malicious script schemes like 'javascript:', 'jscript:', 'vbscript:'.
# Case-insensitive matching.
_MALICIOUS_EXPLICIT_SCHEMES_PATTERN = re.compile(
    r'(javascript|jscript|vbscript):', re.IGNORECASE
)

# Matches the 'data:' scheme. Case-insensitive.
_DATA_SCHEME_PATTERN = re.compile(r'data:', re.IGNORECASE)

# Matches 'data:image/' schemes, which are considered safe. Case-insensitive.
_IMAGE_DATA_SCHEME_PATTERN = re.compile(r'data:image/', re.IGNORECASE)


def is_javascript_scheme(s: str) -> bool:
    """
    Determines if a given string contains a possibly malicious URL scheme
    (such as JavaScript, jscript, vbscript, etc.), with exceptions for
    image data URLs.

    Args:
        s: str, the string to be checked for possibly malicious URL schemes.

    Returns:
        bool: True if the string contains a possibly malicious URL scheme
              (excluding image data URLs), False otherwise.

    Note:
        Improper input (e.g., non-string type for s) may lead to
        exceptions like TypeError from the re module functions.
        Faulty regex patterns could raise re.error during compilation.
    """
    # The prompt allows for exceptions on improper input, so explicit type checking
    # here is more for robustness if desired, but not strictly required by the prompt.
    # `re` functions will raise TypeError if `s` is not a string or bytes-like object.

    # The prompt allows for exceptions on improper input, so explicit type checking
    # here is more for robustness if desired, but not strictly required by the prompt.
    # `re` functions will raise TypeError if `s` is not a string or bytes-like object.

    # Guard clause for explicitly malicious schemes (e.g., javascript:, jscript:, vbscript:)
    if _MALICIOUS_EXPLICIT_SCHEMES_PATTERN.search(s):
        return True

    # Guard clause for 'data:' schemes that are not 'data:image/'
    # Non-image 'data:' URLs can be a vector for XSS (e.g., data:text/html,...).
    # We check for 'data:' first, then ensure it's not an image.
    if _DATA_SCHEME_PATTERN.search(s) and not _IMAGE_DATA_SCHEME_PATTERN.search(s):
        return True

    # If no malicious schemes are found by the guard clauses, the string is considered safe.
    return False
