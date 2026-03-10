"""
Utilities for detecting potentially harmful URL schemes in strings.
"""

import logging
import re

# Module logger for tracking detections.
logger = logging.getLogger(__name__)

# Precompiled regular expressions for performance and clarity.
# Matches leading whitespace/control characters (U+0000 - U+0020).
_LEADING_CTRL_WS_RE = re.compile(r'^[\x00-\x20]+')

# Matches safe image-based data URLs (e.g., "data:image/png;base64,...").
_SAFE_IMAGE_DATA_RE = re.compile(r'^data:image/', flags=re.IGNORECASE)


def check_javascript_scheme(url_str: str) -> bool:
    """
    Determine whether the input string contains a potentially harmful URL scheme.

    Dangerous schemes considered:
    - javascript:
    - vbscript:
    - data: (except safe image-based data URLs like data:image/*)

    The check is case-insensitive and:
    - Ignores leading whitespace/control characters.
    - Tolerates embedded whitespace/control characters within the scheme name,
      normalizing them during parsing (e.g., "java\\u0000script:" → "javascript:").

    Parameters:
        url_str (str): The string to examine for dangerous URL schemes.

    Returns:
        bool: True if a dangerous URL scheme is detected (excluding data:image/*);
              False otherwise.

    Logging:
        Emits a WARNING log when a potentially dangerous scheme is detected,
        including the normalized scheme name.

    Notes:
        Although this function does not intentionally raise exceptions, incorrectly
        formatted inputs or regex patterns could raise 're.error'.
    """
    # Guard: Only process strings; non-strings are treated as safe.
    if not isinstance(url_str, str):
        return False

    # Guard: Empty strings cannot contain a scheme.
    if not url_str:
        return False

    # Remove leading whitespace and control characters for robust scheme detection.
    trimmed_url = _LEADING_CTRL_WS_RE.sub('', url_str)

    # Guard: allow image-based data URLs.
    if _SAFE_IMAGE_DATA_RE.match(trimmed_url):
        return False

    # Parse and normalize the scheme up to the first colon, ignoring embedded
    # whitespace/control characters. Reject invalid characters before the colon.
    scheme_characters = []
    colon_found = False

    for char in trimmed_url:
        if char == ':':
            colon_found = True
            break

        # Valid scheme characters per RFC 3986: ALPHA / DIGIT / "+" / "-" / "."
        if char.isalnum() or char in '+-.':
            scheme_characters.append(char)
            continue

        # Ignore control/whitespace characters embedded before the colon.
        if ord(char) <= 0x20:
            continue

        # Guard: invalid character before colon → not a valid scheme prefix.
        return False

    # Guard: no colon or no accumulated scheme characters → no scheme found.
    if not colon_found or not scheme_characters:
        return False

    scheme = ''.join(scheme_characters).lower()

    # "data:" is dangerous unless it's an image-based data URL (handled above).
    if scheme == 'data':
        logger.warning("Potentially dangerous URL scheme detected: %s", scheme)
        return True

    # Explicitly dangerous scripting schemes.
    if scheme in ('javascript', 'vbscript'):
        logger.warning("Potentially dangerous URL scheme detected: %s", scheme)
        return True

    # All other schemes are considered safe by this checker.
    return False
