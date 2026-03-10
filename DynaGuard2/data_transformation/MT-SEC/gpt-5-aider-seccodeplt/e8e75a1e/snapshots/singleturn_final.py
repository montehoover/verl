import re

# Precompiled regexes for performance and clarity
# Matches dangerous schemes like javascript:, vbscript:, file:, filesystem:
# Ensures the match occurs at start or after a non-scheme character to avoid false positives in words.
_DANGEROUS_SCHEMES_RE = re.compile(
    r'(?:^|[^a-z0-9+.\-])(javascript|vbscript|file|filesystem)\s*:',
    re.IGNORECASE
)

# Matches data: URLs that are NOT image data (e.g., data:text/html, ... or data:;base64,...)
# This ignores valid image data URLs like data:image/png;base64,...
_NON_IMAGE_DATA_URL_RE = re.compile(
    r'(?:^|[^a-z0-9+.\-])data\s*:\s*(?!image/)',
    re.IGNORECASE | re.DOTALL
)


def detect_malicious_url_scheme(url_string: str) -> bool:
    """
    Determine whether the input string contains a risky URL scheme.

    Dangerous schemes include:
      - javascript:
      - vbscript:
      - file:
      - filesystem:
      - data: (except valid image data URLs, e.g., data:image/png;base64,...)

    Args:
        url_string: The input string to inspect for potentially malicious URL schemes.

    Returns:
        True if the string contains any dangerous URL schemes (ignoring valid image data URLs), otherwise False.

    Notes:
        The function doesn't explicitly raise exceptions, but exceptions such as `re.error` may occur if
        regex patterns are invalid or due to unusual inputs.
    """
    if not isinstance(url_string, str) or not url_string:
        return False

    # Check for explicit dangerous schemes
    if _DANGEROUS_SCHEMES_RE.search(url_string):
        return True

    # Check for data: URLs that are not images
    if _NON_IMAGE_DATA_URL_RE.search(url_string):
        return True

    return False
