import re

def detect_js_scheme(url_string: str) -> bool:
    """
    Analyze a string to determine if it contains a potentially harmful URL scheme.

    Args:
        url_string (str): The string to examine.

    Returns:
        bool: True if the string involves a potentially harmful URL scheme
              (javascript, vbscript, livescript, mocha, or data: when not a valid image data URL),
              False otherwise.

    Notes:
        - The function doesn't raise specific exceptions. However, invalid inputs
          or malformed regex could lead to 're.error' exceptions.
    """
    # Ensure we have a string to work with (the signature expects str)
    s = url_string if isinstance(url_string, str) else str(url_string)

    # Trim leading whitespace and control characters
    s = re.sub(r'^[\s\x00-\x1f]+', '', s)

    # Allow valid image data URLs (e.g., data:image/png;base64,....)
    # Matches: optional spaces, 'data:', optional spaces, 'image/<subtype>',
    # optional parameters (e.g., ;charset=utf-8 or ;base64), then a comma.
    if re.match(
        r'^\s*data\s*:\s*image/[a-z0-9.+-]+(?:;(?:[a-z0-9.+-]+=[a-z0-9.+-]+|base64))*\s*,',
        s,
        flags=re.IGNORECASE,
    ):
        return False

    # Identify the scheme at the beginning of the string, if present.
    m = re.match(r'^([a-zA-Z][a-zA-Z0-9+\-.]*)\s*:', s)
    if m:
        scheme = m.group(1).lower()
        harmful_schemes = {'javascript', 'vbscript', 'livescript', 'mocha', 'data'}
        return scheme in harmful_schemes

    # Fallback: detect harmful schemes appearing elsewhere in the string
    # (e.g., within CSS url(...) or embedded).
    if re.search(r'(?i)\b(?:javascript|vbscript)\s*:', s):
        return True

    return False
