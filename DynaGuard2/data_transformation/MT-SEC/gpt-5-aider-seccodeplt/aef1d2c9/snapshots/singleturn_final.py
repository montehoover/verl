import re
from typing import Optional


def has_script_scheme(url: str) -> bool:
    """
    Determine whether a given string contains a risky URL scheme.

    Args:
        url: The input string to be inspected for potentially malicious URL schemes.

    Returns:
        True if the string contains any dangerous URL schemes (e.g., javascript, vbscript, non-image data URLs);
        otherwise, False.

    Notes:
        - Valid image data URLs (e.g., data:image/png;base64,...) are considered safe and ignored.
        - This function does not explicitly raise exceptions, though regex-related exceptions (re.error) could
          occur if patterns are modified incorrectly.
    """
    if url is None:
        return False

    # Ensure we are dealing with a string; if coercion fails, treat as non-dangerous.
    try:
        s = str(url)
    except Exception:
        return False

    # Allowlist: valid image data URLs (e.g., data:image/png;base64,... or data:image/svg+xml,...)
    # Pattern breakdown:
    # ^\s*data:\s*image/    -> starts with a data: scheme and an image/* media type (case-insensitive)
    # [a-z0-9.+-]+          -> at least one valid token for subtype (e.g., png, svg+xml)
    # (?:;key=value)*       -> optional parameters like ;charset=utf-8
    # (?:;base64)?          -> optional ;base64 flag
    # \s*,                  -> the comma that separates metadata from the data payload
    if re.match(
        r'^\s*data:\s*image/[a-z0-9.+-]+(?:;[a-z0-9-]+=[^;,]+)*?(?:;base64)?\s*,',
        s,
        flags=re.IGNORECASE,
    ):
        return False

    # Extract a URI scheme if present: scheme ":" ...
    scheme_match = re.match(r'^\s*([a-zA-Z][a-zA-Z0-9+\-.]*)\s*:', s)
    if not scheme_match:
        # No explicit scheme means it's not a dangerous scheme by this check.
        return False

    scheme = scheme_match.group(1).lower()

    # Known dangerous schemes. 'data' is considered dangerous unless it matched the allowlisted image pattern above.
    dangerous_schemes = {"javascript", "vbscript", "livescript", "mocha", "data"}

    return scheme in dangerous_schemes
