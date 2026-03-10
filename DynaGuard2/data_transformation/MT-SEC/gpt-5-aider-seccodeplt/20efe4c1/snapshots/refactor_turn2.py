import re

_SCHEME_RE = re.compile(r'^\s*([a-zA-Z][a-zA-Z0-9+\-.]*)\s*:\s*', re.IGNORECASE)
_HARMFUL_SCHEMES = {'javascript', 'vbscript', 'livescript', 'mocha'}


def verify_javascript_protocol(url_input: str) -> bool:
    """
    Analyze a given string to determine if it contains a potentially harmful URL scheme,
    such as javascript, vbscript, or similar, while ignoring image data URLs.

    Args:
        url_input: The string to examine for possible dangerous URL schemes.

    Returns:
        True if the string involves a potentially harmful URL scheme, aside from valid image
        data URLs; False otherwise.

    Notes:
        The function doesn't raise specific exceptions. However, invalid inputs or malformed
        regex could lead to 're.error' exceptions.
    """
    if not isinstance(url_input, str):
        return False

    m = _SCHEME_RE.match(url_input)
    if not m:
        return False

    scheme = m.group(1).lower()

    # Immediate guard for common harmful schemes
    if scheme in _HARMFUL_SCHEMES:
        return True

    # If it's not a data URL, and not in harmful schemes, consider it safe
    if scheme != 'data':
        return False

    # Handle data: URLs specially — allow image/* mediatypes
    remainder = url_input[m.end():].lstrip()
    mediatype_segment = remainder.split(',', 1)[0]
    primary_type = mediatype_segment.strip().split(';', 1)[0].strip().lower()

    if primary_type.startswith('image/'):
        return False  # image data URLs are allowed

    # Non-image data URLs are considered potentially harmful
    return True
