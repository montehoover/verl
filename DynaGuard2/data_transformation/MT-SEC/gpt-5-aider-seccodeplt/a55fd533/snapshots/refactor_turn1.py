import re

# Precompiled regex patterns for performance
_JS_RE = re.compile(r'(?<![a-z0-9+.\-])javascript\s*:', re.IGNORECASE)
_VBS_RE = re.compile(r'(?<![a-z0-9+.\-])vbscript\s*:', re.IGNORECASE)
_DATA_RE = re.compile(r'(?<![a-z0-9+.\-])data\s*:', re.IGNORECASE)

# Matches a valid image data URL starting at the "data" occurrence:
# data:image/<subtype>[;charset=...][;base64],...
_SAFE_IMAGE_DATA_RE = re.compile(
    r'\A\s*data\s*:\s*image/(?:png|gif|jpe?g|webp|bmp|svg\+xml)'
    r'(?:;charset=[a-z0-9._\-]+)?'
    r'(?:;base64)?'
    r'\s*,',
    re.IGNORECASE | re.DOTALL,
)


def contains_dangerous_scheme(user_input: str) -> bool:
    """
    Determine whether a string contains a dangerous URL scheme.

    Dangerous schemes considered:
    - javascript:
    - vbscript:
    - data: (except valid image data URLs)

    Args:
        user_input: The input string to inspect.

    Returns:
        True if any dangerous scheme is found (ignoring valid image data URLs), otherwise False.

    Notes:
        This function does not explicitly raise exceptions, but exceptions from the 're'
        module (e.g., re.error) could be raised if regex patterns are incorrect.
    """
    if not isinstance(user_input, str) or not user_input:
        return False

    # Quick checks for script schemes
    if _JS_RE.search(user_input):
        return True
    if _VBS_RE.search(user_input):
        return True

    # Check data: occurrences; flag as dangerous unless it's a valid image data URL
    for m in _DATA_RE.finditer(user_input):
        # Test the substring starting at this 'data' occurrence
        if not _SAFE_IMAGE_DATA_RE.match(user_input[m.start():]):
            return True

    return False
