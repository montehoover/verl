import re

def analyze_scheme(url: str) -> bool:
    """
    Analyze a string to determine if it contains a potentially harmful URL scheme.

    Harmful schemes include:
    - javascript:
    - vbscript:
    - data: (except when the mediatype is an image/* data URL)

    The function ignores valid image data URLs (e.g., data:image/png;base64,...).

    Args:
        url (str): The string to examine for possible dangerous URL schemes.

    Returns:
        bool: True if the string involves a potentially harmful URL scheme (aside from valid
              image data URLs); False otherwise.

    Notes:
        - The function does not intentionally raise specific exceptions. However, invalid inputs
          or malformed regex could lead to re.error exceptions.
    """
    if not isinstance(url, str):
        return False

    s = url.strip()

    # Handle common CSS url(...) wrappers
    m = re.match(r'^url\(\s*(.*?)\s*\)\s*$', s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(1)

    # Strip surrounding quotes if present
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()

    # Remove leading whitespace/control characters
    s = re.sub(r'^[\s\x00-\x1f]+', '', s)

    # Extract scheme (RFC 3986-like), allow optional whitespace before colon to be defensive
    scheme_match = re.match(r'^([a-z][a-z0-9+\-.]*)\s*:', s, flags=re.IGNORECASE)
    if not scheme_match:
        return False

    scheme = scheme_match.group(1).lower()

    if scheme in ('javascript', 'vbscript'):
        return True

    if scheme == 'data':
        # Everything after the first colon
        after = s[scheme_match.end():].lstrip()

        # Allow only image/* data URLs to be considered safe
        # Valid forms include:
        # - image/png;base64,....
        # - image/svg+xml;utf8,....
        # - image/jpeg,...
        if re.match(r'(?i)^image/[a-z0-9.+-]+(?:;|,|$)', after):
            return False
        return True

    # Other schemes are treated as not harmful by this function.
    return False
