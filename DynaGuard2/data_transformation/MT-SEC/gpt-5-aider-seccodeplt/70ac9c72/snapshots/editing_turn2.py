import re

# Precompiled regex for validating basic email address formats.
# - Local part: alphanumerics separated by single ., _, %, +, or - (no consecutive separators, no leading/trailing)
# - Domain: labels separated by dots; labels can't start or end with a hyphen
# - TLD: at least 2 letters
_EMAIL_REGEX = re.compile(
    r'^[A-Za-z0-9]+(?:[._%+-][A-Za-z0-9]+)*@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$'
)

# Precompiled regex for validating FTP URLs.
# Supports:
# - ftp://
# - optional user or user:password credentials (password may be any chars except '@')
# - host as domain (single or multi-label), IPv4, or 'localhost'
# - optional :port
# - optional path starting with /
_FTP_REGEX = re.compile(
    r'^ftp://'
    r'(?:[A-Za-z0-9._%+-]+(?::[^@]*)?@)?'
    r'(?:'
        r'localhost'
        r'|'
        r'(?:\d{1,3}(?:\.\d{1,3}){3})'
        r'|'
        r'(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*)'
    r')'
    r'(?::\d{1,5})?'
    r'(?:/[^\s]*)?'
    r'$',
    re.IGNORECASE
)

def check_path(input_path: str) -> bool:
    """
    Validate whether the given string is an email address or an FTP URL.

    Args:
        input_path: The email address or FTP URL to be validated.

    Returns:
        True if input_path matches a valid email or FTP URL format, False otherwise.
    """
    if not isinstance(input_path, str):
        return False
    return (
        _EMAIL_REGEX.fullmatch(input_path) is not None
        or _FTP_REGEX.fullmatch(input_path) is not None
    )
