import re

# Compiled regular expression to validate HTTP/HTTPS URLs
_HTTP_URL_RE = re.compile(
    r"""
    ^
    https?://
    (
        localhost
        |
        (?:                                     # domain
            (?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+
            [A-Za-z]{2,}
        )
        |
        (?:                                     # IPv4
            (?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)
            (?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}
        )
    )
    (?::\d{1,5})?
    (?:/[^\s?#]*)?           # path
    (?:\?[^\s#]*)?           # query
    (?:\#[^\s]*)?            # fragment
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)


def validate_path(path: str) -> bool:
    """
    Validate that the given string is a valid HTTP or HTTPS URL.

    Args:
        path (str): The input path to be validated.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        if not isinstance(path, str):
            return False
        candidate = path.strip()
        if not candidate:
            return False
        return _HTTP_URL_RE.fullmatch(candidate) is not None
    except Exception:
        # Ensure no exceptions are raised per requirements
        return False
