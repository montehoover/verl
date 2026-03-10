import re

# Predefined regular expression to validate http/https URLs with optional path, query, and fragment.
_HTTP_URL_RE = re.compile(
    r"""
    ^https?://
    (                                   # Host alternatives:
        localhost
        |
        (?:                             # Domain name (labels)
            [A-Za-z0-9]                 # start char
            (?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?
            (?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*
            \.[A-Za-z]{2,63}            # TLD
        )
        |
        (?:                             # IPv4
            (?:25[0-5]|2[0-4]\d|1?\d{1,2})
            (?:\.(?:25[0-5]|2[0-4]\d|1?\d{1,2})){3}
        )
    )
    (?::\d{1,5})?                       # Optional port
    (?:/[^\s?#]*)?                      # Optional path
    (?:\?[^\s#]*)?                      # Optional query
    (?:#[^\s]*)?                        # Optional fragment
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def path_check(u: str) -> bool:
    """
    Check whether a given string represents a valid http or https path/URL.

    Args:
        u: str - the input string to validate.

    Returns:
        True if the input matches the predefined http/https URL pattern, else False.
    """
    try:
        if not isinstance(u, str):
            return False
        match = _HTTP_URL_RE.fullmatch(u)
        return match is not None
    except Exception:
        # Do not raise exceptions; return False on any unexpected error.
        return False
