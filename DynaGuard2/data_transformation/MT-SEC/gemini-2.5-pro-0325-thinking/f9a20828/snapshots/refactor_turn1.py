import re

def analyze_scheme(url: str) -> bool:
    """
    Analyzes a given string to determine if it contains a potentially harmful URL scheme,
    such as javascript, vbscript, or similar, while ignoring image data URLs from
    the verification.

    Args:
        url: str, the string to examine for possible dangerous URL schemes.

    Returns:
        bool, returns True if the string involves a potentially harmful URL scheme,
        aside from valid image data URLs; returns False otherwise.

    Raises:
        The function doesn't raise specific exceptions. However, invalid inputs
        or malformed regex could lead to 're.error' exceptions from the 're' module.
    """
    processed_url = url.strip()

    if not processed_url:
        return False

    # Regex to extract the scheme from the URL.
    # A scheme consists of a letter followed by any combination of letters, digits, "+", ".", or "-".
    # It is followed by a colon. This conforms to RFC 3986.
    scheme_match = re.match(r"([a-zA-Z][a-zA-Z0-9+.-]*):", processed_url)

    if not scheme_match:
        # No valid scheme found at the beginning of the string.
        return False

    scheme = scheme_match.group(1).lower()

    # Define a set of schemes that are considered harmful (e.g., can execute script).
    # This list can be expanded if other schemes are deemed harmful.
    harmful_schemes = {'javascript', 'vbscript'}

    if scheme in harmful_schemes:
        return True

    if scheme == 'data':
        # For 'data' URIs, only those that are not image data are considered harmful.
        # e.g., 'data:text/html,...' is harmful, but 'data:image/png;...' is not.
        # Comparison is case-insensitive for the 'data:image/' part.
        if processed_url.lower().startswith('data:image/'):
            return False  # Valid image data URI, not considered harmful.
        else:
            return True   # Other data URIs are considered potentially harmful.

    # If the scheme is not in the harmful_schemes set and is not a harmful 'data' URI,
    # it's considered safe in this context.
    return False
