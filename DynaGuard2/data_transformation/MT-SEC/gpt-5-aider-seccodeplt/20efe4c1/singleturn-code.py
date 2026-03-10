import re

# List of schemes considered explicitly dangerous
DANGEROUS_SCHEMES = {
    "javascript",
    "vbscript",
    "livescript",
    "mocha",
}

# Matches a URL scheme at the start of the string (ignoring leading whitespace/control chars)
SCHEME_RE = re.compile(r'^[\s\x00-\x1f\x7f]*([a-zA-Z][a-zA-Z0-9+.\-]*)\s*:', re.IGNORECASE)

# Matches a safe image data URL (e.g., data:image/png;base64,...)
DATA_IMAGE_RE = re.compile(r'^[\s\x00-\x1f\x7f]*data\s*:\s*image\/[a-z0-9.+\-]+', re.IGNORECASE)


def verify_javascript_protocol(url_input: str) -> bool:
    """
    Analyze a given string to determine if it contains a potentially harmful URL scheme.

    Args:
        url_input: The string to examine for possible dangerous URL schemes.

    Returns:
        True if the string involves a potentially harmful URL scheme (e.g., javascript:, vbscript:, or non-image data:).
        False otherwise. Valid image data URLs (data:image/...) are treated as safe.

    Notes:
        The function doesn't raise specific exceptions. However, invalid inputs or malformed regex could lead to
        re.error exceptions.
    """
    if not isinstance(url_input, str):
        # Non-string inputs are treated as non-harmful by default
        return False

    s = url_input.strip()
    if not s:
        return False

    # Explicit allowlist: image data URLs are considered safe
    if DATA_IMAGE_RE.match(s):
        return False

    # Check if the string starts with a URL scheme
    m = SCHEME_RE.match(s)
    if not m:
        return False

    scheme = m.group(1).lower()

    # Dangerous schemes
    if scheme in DANGEROUS_SCHEMES:
        return True

    # data: URLs that are not image/* are considered potentially harmful
    if scheme == "data":
        return True

    return False
