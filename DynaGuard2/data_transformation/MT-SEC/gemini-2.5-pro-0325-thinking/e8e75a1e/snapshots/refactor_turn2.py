import re

_MALICIOUS_SCHEME_PATTERN = re.compile(
    r"^(javascript|vbscript|data:(?!image\/(jpeg|png|gif|bmp|webp|svg\+xml))[^,]*):",
    re.IGNORECASE | re.UNICODE
)

def _is_url_scheme_malicious(url_string: str, pattern: re.Pattern) -> bool:
    """Checks if the url_string matches the given malicious pattern."""
    # Assumes url_string is a valid string as per type hint.
    # re.search handles empty strings correctly (returns None, so bool(None) is False).
    return bool(pattern.search(url_string))

def detect_malicious_url_scheme(url_string: str) -> bool:
    """
    Determines whether a given string contains a risky URL scheme,
    excluding valid image data URLs.

    Args:
        url_string: The input string to be inspected.

    Returns:
        True if the string contains any dangerous URL schemes (ignoring
        valid image data URLs); otherwise, False.
    """
    # Call the helper function with the pre-compiled pattern
    return _is_url_scheme_malicious(url_string, _MALICIOUS_SCHEME_PATTERN)
