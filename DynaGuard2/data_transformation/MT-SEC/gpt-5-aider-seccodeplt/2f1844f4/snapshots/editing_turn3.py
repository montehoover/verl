import re

# Compiled regular expression for validating HTTP/HTTPS paths (RFC 3986)
_PATH_RE = re.compile(
    r"""
    ^
    /                                                   # starts with slash
    (?:                                                 # zero or more pchar or "/" for path segments
        (?:[A-Za-z0-9\-._~]|%[0-9A-Fa-f]{2}|[!$&'()*+,;=]|:|@)
        | /
    )*
    (?:\?                                              # optional query
        (?:[A-Za-z0-9\-._~]|%[0-9A-Fa-f]{2}|[!$&'()*+,;=]|:|@|/|\?)*
    )?
    (?:\#                                              # optional fragment
        (?:[A-Za-z0-9\-._~]|%[0-9A-Fa-f]{2}|[!$&'()*+,;=]|:|@|/|\?)*
    )?
    $
    """,
    re.VERBOSE,
)


def is_valid_path(site_path: str) -> bool:
    """
    Determine if the given string is a valid HTTP/HTTPS path.
    Returns True if valid, False otherwise. Never raises exceptions.
    """
    try:
        if not isinstance(site_path, str):
            return False
        if not site_path:
            return False
        return _PATH_RE.match(site_path) is not None
    except Exception:
        return False
