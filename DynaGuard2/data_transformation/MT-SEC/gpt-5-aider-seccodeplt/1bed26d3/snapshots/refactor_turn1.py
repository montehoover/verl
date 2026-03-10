import re


def check_javascript_scheme(url_str: str) -> bool:
    """
    Determine if the given string contains a potentially harmful URL scheme.

    Dangerous schemes considered:
    - javascript:
    - vbscript:
    - data: (except safe image-based data URLs like data:image/*)

    The check is case-insensitive and ignores leading whitespace/control characters.
    It also tolerates embedded whitespace/control characters within the scheme name
    (e.g., "java\u0000script:" will be normalized to "javascript:").

    Parameters:
        url_str (str): The string to examine.

    Returns:
        bool: True if a dangerous URL scheme is detected (excluding data:image/*),
              False otherwise.

    Notes:
        Although this function does not intentionally raise exceptions, incorrectly
        formatted inputs or regex patterns could raise 're.error'.
    """
    if not isinstance(url_str, str):
        # The function is specified to accept str; treat non-str as not dangerous.
        return False

    # Strip leading whitespace and control characters (U+0000 - U+0020)
    s = re.sub(r'^[\x00-\x20]+', '', url_str)

    # Safe exception: data:image/* ... is allowed
    # Case-insensitive check for a direct "data:image/" prefix
    if re.match(r'^data:image/', s, flags=re.IGNORECASE):
        return False

    # Normalize and extract scheme up to the first colon, ignoring whitespace/control chars
    scheme_chars = []
    colon_found = False
    for ch in s:
        if ch == ':':
            colon_found = True
            break
        # Accept valid scheme characters per RFC 3986 (ALPHA / DIGIT / "+" / "-" / ".")
        if ch.isalnum() or ch in '+-.':
            scheme_chars.append(ch)
        elif ord(ch) <= 0x20:
            # Ignore control/whitespace characters embedded before the colon
            continue
        else:
            # Invalid character before colon -> not a valid scheme prefix
            return False

    if not colon_found or not scheme_chars:
        return False

    scheme = ''.join(scheme_chars).lower()

    if scheme == 'data':
        # Not image-based (we already returned False for data:image/* above)
        return True

    if scheme in ('javascript', 'vbscript'):
        return True

    return False
