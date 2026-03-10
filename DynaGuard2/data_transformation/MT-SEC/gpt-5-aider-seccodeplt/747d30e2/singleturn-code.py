import re

# Precompiled regex patterns for performance and clarity
_SCHEME_RE = re.compile(r'^[\s\x00-\x1f]*([a-z][a-z0-9+.\-]*)\s*:', re.IGNORECASE | re.DOTALL)
_DATA_PREFIX_RE = re.compile(r'^[\s\x00-\x1f]*data:', re.IGNORECASE | re.DOTALL)
# Capture media type up to the first comma, allowing for optional parameters (e.g., ;base64)
_DATA_MIME_RE = re.compile(r'^[\s\x00-\x1f]*data:([^;,]*)(?:;[^,]*)?,', re.IGNORECASE | re.DOTALL)

# Commonly abused schemes
_BAD_SCHEMES = {
    'javascript',
    'jscript',
    'vbscript',
    'livescript',
    'mocha',
    # Treat non-image data: as malicious via explicit handling below
    'data',
}


def is_javascript_scheme(s: str) -> bool:
    """
    Check if the given string contains a possibly malicious URL scheme.

    - Returns True for schemes like javascript:, jscript:, vbscript:, etc.
    - Returns True for data: URLs that are not image data URLs (e.g., data:text/html,...).
    - Returns False for image data URLs (data:image/...).
    - Ignores leading whitespace and control characters.

    Note: This function does not explicitly raise exceptions, but invalid inputs or
    faulty regex patterns could raise exceptions such as re.error.
    """
    if not isinstance(s, str):
        # Not explicitly raising; treat non-string as non-malicious
        return False

    # Handle data: scheme separately to exempt image data URLs
    if _DATA_PREFIX_RE.match(s):
        m = _DATA_MIME_RE.match(s)
        if not m:
            # data: without a comma-delimited payload is suspicious
            return True
        media_type = (m.group(1) or '').strip().lower()
        if media_type == '':
            # No explicit media type (e.g., "data:;base64,...") -> treat as malicious
            return True
        # Exempt only image media types (e.g., image/png, image/svg+xml)
        return not media_type.startswith('image/')

    # Check for other potentially malicious schemes at the beginning of the string
    m = _SCHEME_RE.match(s)
    if not m:
        return False

    scheme = m.group(1).lower()
    return scheme in _BAD_SCHEMES
