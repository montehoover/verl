import re

# Regular expression to validate HTTP/HTTPS URLs.
_URL_RE = re.compile(
    r"""^
    https?://
    (?:
        # IPv4 address
        (?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)
        |
        # Hostname (including subdomains)
        (?:(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)\.)+(?:[A-Za-z]{2,})
        |
        # localhost
        localhost
    )
    (?::\d{1,5})?
    (?:/[A-Za-z0-9._~!$&'()*+,;=:@%\-]*)*
    (?:\?[A-Za-z0-9._~!$&'()*+,;=:@%/\-]*)?
    (?:\#[A-Za-z0-9._~!$&'()*+,;=:@%/\-]*)?
    $
    """,
    re.VERBOSE,
)

def is_valid_path(site_path: str) -> bool:
    """
    Determines if a given string is a valid http or https path (URL) by matching
    against a regular expression.

    Args:
        site_path: The string to evaluate as a potential valid path.

    Returns:
        True if the string is a valid http/https path, otherwise False.

    Notes:
        No exceptions are raised by this function.
    """
    try:
        if not isinstance(site_path, str):
            return False
        candidate = site_path.strip()
        if not candidate:
            return False
        return _URL_RE.match(candidate) is not None
    except Exception:
        # Ensure no exceptions escape this function.
        return False
