import re

_SCHEME_RE = re.compile(r'^\s*([a-zA-Z][a-zA-Z0-9+\-.]*)\s*:\s*', re.IGNORECASE)


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

    # Handle data: URLs specially — allow image/* mediatypes
    if scheme == 'data':
        remainder = url_input[m.end():].lstrip()
        # Extract mediatype segment (before the first comma)
        mediatype_segment = remainder.split(',', 1)[0]
        # Extract the primary type before any parameters
        primary_type = mediatype_segment.strip().split(';', 1)[0].strip().lower()
        if primary_type.startswith('image/'):
            return False  # image data URLs are allowed
        return True  # non-image data URLs are considered potentially harmful

    # Common script-based or similar harmful schemes
    harmful_schemes = {'javascript', 'vbscript', 'livescript', 'mocha'}

    return scheme in harmful_schemes
