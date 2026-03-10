"""Utility for detecting potentially malicious URL schemes within strings.

This module exposes a single function, is_javascript_scheme, which inspects
an input string for potentially dangerous URL schemes commonly used in XSS
payloads (e.g., javascript:, jscript:, vbscript:, livescript:). It also flags
data: URLs except when the payload is an image (e.g., data:image/png;...).
"""

import re


# Matches scripting URL schemes such as:
# - javascript:
# - jscript:
# - vbscript:
# - livescript:
#
# Regex breakdown:
# - \b ensures we start at a word boundary, so we don't match inside other words.
# - (?:javascript|jscript|vbscript|livescript) matches any of the listed schemes.
# - \s*: allows optional whitespace before the required colon.
SCRIPT_SCHEME_RE = re.compile(
    r"\b(?:javascript|jscript|vbscript|livescript)\s*:",
    flags=re.IGNORECASE,
)


# Matches data: URLs that are not images, e.g., data:text/html,... or data:application/pdf,...
#
# Regex breakdown:
# - \bdata\s*: matches the "data" scheme with optional whitespace before the colon.
# - :\s*: allows optional whitespace after the colon.
# - (?!image/): negative lookahead that ensures the media type does NOT start with "image/".
DATA_NON_IMAGE_RE = re.compile(
    r"\bdata\s*:\s*(?!image/)",
    flags=re.IGNORECASE,
)


def is_javascript_scheme(s: str) -> bool:
    """
    Determine whether the input string contains a potentially malicious URL scheme.

    The function returns True if:
    - A scripting scheme is present (javascript:, jscript:, vbscript:, or livescript:),
      case-insensitively, allowing optional whitespace before the colon.
    - A data: URL is present where the MIME type does not begin with "image/"
      (e.g., data:text/html, data:application/pdf), allowing optional whitespace
      around the colon.

    Image data URLs (e.g., data:image/png;...) are explicitly excluded and will
    not trigger a True result.

    Parameters:
        s (str): The string to scan for potentially malicious URL schemes.

    Returns:
        bool: True if a possibly malicious scheme is found (excluding image data URLs),
              otherwise False.
    """
    # Search for any scripting scheme occurrence
    if SCRIPT_SCHEME_RE.search(s):
        return True

    # Search for any non-image data: URL occurrence
    if DATA_NON_IMAGE_RE.search(s):
        return True

    return False
