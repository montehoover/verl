import re

# Regex to capture a leading URL scheme (e.g., "http:", "javascript:")
_SCHEME_RE = re.compile(
    r'^\s*([a-zA-Z][a-zA-Z0-9+\-.]*)\s*:\s*',
    re.IGNORECASE,
)

# Set of known script-based or otherwise harmful schemes
_HARMFUL_SCHEMES = {
    'javascript',
    'vbscript',
    'livescript',
    'mocha',
}


def verify_javascript_protocol(url_input: str) -> bool:
    """
    Return True if url_input uses a potentially harmful URL scheme.

    Recognizes common script schemes (e.g., javascript, vbscript). Allows
    data:image/* URLs, but flags other data: URLs. Returns False for
    non-strings, strings without a scheme, or safe schemes.

    Args:
        url_input: The string to examine.

    Returns:
        bool: True if a harmful scheme is detected (excluding image data
        URLs), otherwise False.

    Note:
        This function does not catch exceptions from the regex engine.
        Malformed patterns or certain invalid inputs could raise re.error.
    """
    # Guard: Only process strings; treat non-string inputs as safe.
    if not isinstance(url_input, str):
        return False

    # Guard: If no scheme prefix is present, it's considered safe.
    match = _SCHEME_RE.match(url_input)
    if not match:
        return False

    # Normalize the extracted scheme for comparison.
    scheme = match.group(1).lower()

    # Guard: Immediately flag known harmful schemes (e.g., javascript).
    if scheme in _HARMFUL_SCHEMES:
        return True

    # Guard: If not a data: URL, and not harmful, consider it safe.
    if scheme != 'data':
        return False

    # Handle data: URLs — allow image/* media types, flag others.
    # The data URL format is: data:[<mediatype>][;base64],<data>
    # Split on the first comma to isolate the mediatype segment.
    remainder = url_input[match.end():].lstrip()
    mediatype_segment = remainder.split(',', 1)[0]

    # Extract the primary type before any parameters (e.g., image/svg+xml).
    primary_type = (
        mediatype_segment.strip().split(';', 1)[0].strip().lower()
    )

    # Allow image/* data URLs.
    if primary_type.startswith('image/'):
        return False  # image data URLs are allowed

    # Non-image data URLs are considered potentially harmful.
    return True
