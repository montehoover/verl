import re
from typing import Pattern, Iterator, Tuple

# This module provides detect_malicious_url_scheme(url_string: str) -> bool
# The function returns True if the input string contains a risky URL scheme,
# such as "javascript:", "vbscript:", "data:" (except valid image data URLs),
# "file:", or "filesystem:". It ignores valid image data URLs (e.g.,
# "data:image/png;base64,...").
#
# Notes:
# - The function does not explicitly raise exceptions; however, re.error could
#   be raised if regex patterns are invalid (they are static and valid here).
# - Scheme detection is case-insensitive.
# - "data:" is treated as dangerous unless it is a valid image data URL.


# Regex to find any URI scheme occurrences like "<scheme>:"
_SCHEME_RE: Pattern[str] = re.compile(r'([a-z][a-z0-9+.\-]{0,31}):', re.IGNORECASE)

# Regex to recognize a valid image data URL, starting at "data:" (case-insensitive).
# Allows optional whitespace after "data:", standard image media types, optional
# parameters (e.g., ;charset=utf-8), optional ;base64, followed by a comma and data.
_VALID_IMAGE_DATA_URL_RE: Pattern[str] = re.compile(
    r'^data:\s*image/[a-z0-9.+\-]+(?:;[a-z0-9.+\-]+=[a-z0-9.+/\+=\-]*)*(?:;base64)?,',
    re.IGNORECASE,
)

# Known dangerous schemes. "data" is conditionally dangerous; see handling below.
_DANGEROUS_SCHEMES = {
    'javascript',
    'vbscript',
    'data',        # handled specially to allow valid image data URLs
    'file',
    'filesystem',
}


def _iter_scheme_occurrences(text: str) -> Iterator[Tuple[str, int]]:
    """
    Pure helper to extract URI scheme occurrences.

    Yields:
        Tuples of (scheme_lowercase, start_index) for each "<scheme>:" occurrence.
    """
    for match in _SCHEME_RE.finditer(text):
        yield match.group(1).lower(), match.start()


def detect_malicious_url_scheme(url_string: str) -> bool:
    """
    Determine whether the given string contains a risky URL scheme.

    A risky scheme is one of: javascript:, vbscript:, data:, file:, filesystem:
    The data: scheme is considered safe only when it is a valid image data URL
    (e.g., data:image/png;base64,...). Other schemes are flagged as risky.

    Args:
        url_string: The input string to inspect.

    Returns:
        True if any dangerous URL scheme is found (ignoring valid image data URLs); otherwise, False.

    Raises:
        This function does not explicitly raise exceptions. However, exceptions such as re.error
        could occur due to regex engine errors.
    """
    if not url_string:
        return False

    for scheme, start_idx in _iter_scheme_occurrences(url_string):
        # Skip non-dangerous schemes early (guard clause)
        if scheme not in _DANGEROUS_SCHEMES:
            continue

        # Handle data: specially — allow valid image data URLs
        if scheme == 'data':
            remainder = url_string[start_idx:]
            if _VALID_IMAGE_DATA_URL_RE.match(remainder):
                continue
            return True

        # All other dangerous schemes are malicious
        return True

    return False
